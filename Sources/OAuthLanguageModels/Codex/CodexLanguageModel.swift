import AnyLanguageModel
import Foundation

public let defaultCodexResponsesBaseURL = URL(string: "https://chatgpt.com/backend-api/")!

// MARK: - CodexLanguageModel

/// `LanguageModel` implementation that talks to ChatGPT / Codex over the
/// undocumented `/backend-api/codex/responses` SSE endpoint, authenticated
/// with a ChatGPT account access token (typically obtained via
/// `CodexOAuthFlow`).
///
/// Drop into any `LanguageModelSession` from AnyLanguageModel:
///
/// ```swift
/// let model = CodexLanguageModel(
///     tokenProvider: { try await myAuth.validToken() },
///     model: "gpt-5"
/// )
/// let session = LanguageModelSession(model: model)
/// ```
public struct CodexLanguageModel: LanguageModel {
    // MARK: Lifecycle

    public init(
        tokenProvider: @escaping @Sendable () async throws -> CodexToken,
        model: String,
        baseURL: URL = defaultCodexResponsesBaseURL,
        sessionID: String = UUID().uuidString.lowercased(),
        originator: String = "OAuthLanguageModels"
    ) {
        self.tokenProvider = tokenProvider
        self.model = model
        self.baseURL = baseURL
        self.sessionID = sessionID
        self.originator = originator
        state = CodexSessionState()
    }

    // MARK: Public

    public typealias UnavailableReason = Never

    public let tokenProvider: @Sendable () async throws -> CodexToken
    public let model: String
    public let baseURL: URL
    public let sessionID: String
    /// Identifies the client to OpenAI. Sent as the `originator` HTTP header
    /// and audited server-side.
    public let originator: String

    public func respond<Content: Generable>(
        within session: LanguageModelSession,
        to _: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt _: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        guard type == String.self else {
            throw CodexLanguageModelError.unsupportedContentType
        }

        let custom = options[custom: Self.self] ?? .init()
        var inputs = try await buildInputs(from: session.transcript)
        let tools = session.tools.map(Self.convertToolToOpenResponsesFormat)
        var entries: [Transcript.Entry] = []

        while true {
            let response = try await send(
                inputs: inputs,
                instructions: session.instructions?.description,
                tools: tools.isEmpty ? nil : tools,
                options: options,
                custom: custom
            )

            // Reasoning items appear in `output` ahead of any
            // `function_call` / `message` items in the same turn. They
            // must be replayed in subsequent requests within this same
            // tool-loop so the model retains its chain of thought across
            // rounds (required because `store: false`).
            inputs.append(contentsOf: response.reasoningItems)

            let toolCalls = response.toolCalls
            if !toolCalls.isEmpty {
                await state.remember(toolCalls)
                inputs.append(contentsOf: Self.makeFunctionCallItems(for: toolCalls))

                let resolution = try await resolveToolCalls(toolCalls, session: session)
                switch resolution {
                case let .stop(calls):
                    if !calls.isEmpty {
                        entries.append(.toolCalls(Transcript.ToolCalls(calls)))
                    }
                    let empty = try emptyResponseContent(for: type)
                    return LanguageModelSession.Response(
                        content: empty.content,
                        rawContent: empty.rawContent,
                        transcriptEntries: ArraySlice(entries)
                    )
                case let .invocations(invocations):
                    if !invocations.isEmpty {
                        entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                        for invocation in invocations {
                            entries.append(.toolOutput(invocation.output))
                            inputs.append(Self.makeFunctionCallOutput(for: invocation.output))
                        }
                        continue
                    }
                }
            }

            let text = response.outputText ?? Self.extractText(from: response.output) ?? ""
            guard text.isEmpty == false || response.output != nil else {
                throw CodexLanguageModelError.noResponseGenerated
            }
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(entries)
            )
        }
    }

    public func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init { continuation in
            let task = Task {
                do {
                    let response = try await respond(
                        within: session,
                        to: prompt,
                        generating: type,
                        includeSchemaInPrompt: includeSchemaInPrompt,
                        options: options
                    )
                    continuation.yield(.init(content: response.content.asPartiallyGenerated(), rawContent: response.rawContent))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }

        return LanguageModelSession.ResponseStream(stream: stream)
    }

    // MARK: Fileprivate

    /// Top-level request keys callers may not override via
    /// `CustomGenerationOptions.extraBody`. These either define the
    /// OAuth/Codex request shape, are required for cross-call cache
    /// stability (`prompt_cache_key`), or are required for reasoning
    /// replay correctness (`include`).
    fileprivate static let reservedBodyKeys: Set<String> = [
        "model", "input", "instructions", "tools",
        "prompt_cache_key", "store", "stream", "include"
    ]

    // MARK: Private

    private static var userAgent: String {
        let processInfo = ProcessInfo.processInfo
        let version = processInfo.operatingSystemVersion
        let osName: String
        #if os(macOS)
        osName = "macOS"
        #elseif os(Linux)
        osName = "Linux"
        #else
        osName = "Unknown"
        #endif
        return "OAuthLanguageModels (\(osName) \(version.majorVersion).\(version.minorVersion).\(version.patchVersion))"
    }

    private let state: CodexSessionState

    private static func convertPromptSegments(_ segments: [Transcript.Segment], assistant: Bool) -> [JSONValue] {
        segments.compactMap { segment -> JSONValue? in
            switch segment {
            case let .text(text):
                return JSONValue.object([
                    "type": .string(assistant ? "output_text" : "input_text"),
                    "text": .string(text.content)
                ])
            case let .structure(structured):
                let text: String = switch structured.content.kind {
                case let .string(value):
                    value
                default:
                    structured.content.jsonString
                }
                return JSONValue.object([
                    "type": .string(assistant ? "output_text" : "input_text"),
                    "text": .string(text)
                ])
            case let .image(image):
                switch image.source {
                case let .url(url):
                    return JSONValue.object(["type": .string("input_image"), "image_url": .string(url.absoluteString)])
                case let .data(data, mimeType):
                    return JSONValue.object([
                        "type": .string("input_image"),
                        "image_url": .string("data:\(mimeType);base64,\(data.base64EncodedString())")
                    ])
                }
            }
        }
    }

    private static func makeFunctionCallItems(for calls: [ProviderToolCall]) -> [JSONValue] {
        calls.map { call in
            .object([
                "id": .string(call.itemID ?? call.id),
                "type": .string("function_call"),
                "call_id": .string(call.id),
                "name": .string(call.name),
                "arguments": .string((try? encodedJSONString(for: call.arguments)) ?? "{}")
            ])
        }
    }

    private static func makeFunctionCallOutput(for output: Transcript.ToolOutput) -> JSONValue {
        .object([
            "type": .string("function_call_output"),
            "call_id": .string(providerCallID(fromTranscriptID: output.id)),
            "output": .string(toolOutputString(output.segments))
        ])
    }

    private static func toolOutputString(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case let .text(text):
                text.content
            case let .structure(structured):
                switch structured.content.kind {
                case let .string(value):
                    value
                default:
                    structured.content.jsonString
                }
            case .image:
                nil
            }
        }.joined(separator: "\n")
    }

    private static func convertToolToOpenResponsesFormat(_ tool: any Tool) -> OpenResponsesTool {
        let parameters = try? providerToolSchemaJSONValue(for: tool.parameters)
        return OpenResponsesTool(name: tool.name, description: tool.description, parameters: parameters)
    }

    private static func makeRequestBody(
        model: String,
        instructions: String,
        inputs: [JSONValue],
        tools: [OpenResponsesTool]?,
        promptCacheKey: String,
        options: GenerationOptions,
        custom: CustomGenerationOptions
    ) -> JSONValue {
        let verbosity = custom.verbosity?.rawValue ?? "medium"
        let parallel = custom.parallelToolCalls ?? true

        var body: [String: JSONValue] = [
            "model": .string(model),
            "instructions": .string(instructions),
            "input": .array(inputs),
            "prompt_cache_key": .string(promptCacheKey),
            "store": .bool(false),
            "stream": .bool(true),
            "text": .object(["verbosity": .string(verbosity)]),
            "include": .array([.string("reasoning.encrypted_content")]),
            "parallel_tool_calls": .bool(parallel)
        ]

        if let tools, !tools.isEmpty {
            body["tools"] = .array(tools.map(\.jsonValue))
        }

        // Sampling / generation knobs.
        if let temperature = options.temperature {
            body["temperature"] = .double(temperature)
        }
        if let topP = custom.topP {
            body["top_p"] = .double(topP)
        }
        if let maxOutput = custom.maxOutputTokens ?? options.maximumResponseTokens {
            body["max_output_tokens"] = .int(maxOutput)
        }
        if let maxToolCalls = custom.maxToolCalls {
            body["max_tool_calls"] = .int(maxToolCalls)
        }

        if let reasoning = custom.reasoning {
            var reasoningObject: [String: JSONValue] = [:]
            if let effort = reasoning.effort {
                reasoningObject["effort"] = .string(effort.rawValue)
            }
            if let summary = reasoning.summary {
                reasoningObject["summary"] = .string(summary.rawValue)
            }
            if !reasoningObject.isEmpty {
                body["reasoning"] = .object(reasoningObject)
            }
        }

        // Tool choice (defaults to "auto" if unset).
        body["tool_choice"] = custom.toolChoice.map(toolChoiceJSON) ?? .string("auto")

        // extraBody is merged last and may not override reserved keys.
        if let extra = custom.extraBody {
            for (key, value) in extra where !reservedBodyKeys.contains(key) {
                body[key] = value
            }
        }

        return .object(body)
    }

    private static func toolChoiceJSON(_ choice: CustomGenerationOptions.ToolChoice) -> JSONValue {
        switch choice {
        case .none: return .string("none")
        case .auto: return .string("auto")
        case .required: return .string("required")
        case let .function(name):
            return .object(["type": .string("function"), "name": .string(name)])
        case let .allowedTools(names, mode):
            let descriptors = names.map { name in
                JSONValue.object(["type": .string("function"), "name": .string(name)])
            }
            return .object([
                "type": .string("allowed_tools"),
                "mode": .string(mode.rawValue),
                "tools": .array(descriptors)
            ])
        }
    }

    private static func extractToolCalls(from output: [JSONValue]?) throws -> [ProviderToolCall] {
        guard let output else { return [] }
        var result: [ProviderToolCall] = []
        for item in output {
            collectToolCalls(from: item, into: &result)
        }
        return result
    }

    private static func collectToolCalls(from value: JSONValue, into result: inout [ProviderToolCall]) {
        switch value {
        case let .object(object):
            let type = object["type"].flatMap {
                if case let .string(string) = $0 { string } else { nil }
            }
            if let type, ["function_call", "tool_call", "tool_use"].contains(type),
               let call = try? parseToolCall(from: object) {
                result.append(call)
            }
            if let item = object["item"] {
                collectToolCalls(from: item, into: &result)
            }
            if let toolCall = object["tool_call"] {
                collectToolCalls(from: toolCall, into: &result)
            }
            if let content = object["content"] {
                collectToolCalls(from: content, into: &result)
            }
            for (key, value) in object where key != "content" && key != "item" && key != "tool_call" {
                collectToolCalls(from: value, into: &result)
            }
        case let .array(array):
            for item in array {
                collectToolCalls(from: item, into: &result)
            }
        default:
            break
        }
    }

    private static func parseToolCall(from object: [String: JSONValue]) throws -> ProviderToolCall? {
        let itemID = object["id"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        }
        let callID = object["call_id"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        } ?? itemID
        let name = object["name"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        }
        guard let callID, let name, !callID.isEmpty, !name.isEmpty else { return nil }

        let argumentsJSON: String
        if let arguments = object["arguments"] {
            switch arguments {
            case let .string(string):
                argumentsJSON = string
            case let .object(object):
                let data = try JSONEncoder.deterministic.encode(JSONValue.object(object))
                argumentsJSON = String(data: data, encoding: .utf8) ?? "{}"
            default:
                argumentsJSON = "{}"
            }
        } else {
            argumentsJSON = "{}"
        }

        return try ProviderToolCall(
            id: callID,
            itemID: itemID,
            name: name,
            arguments: GeneratedContent(json: argumentsJSON)
        )
    }

    private static func extractText(from output: [JSONValue]?) -> String? {
        guard let output else { return nil }
        var parts: [String] = []
        for item in output {
            guard case let .object(object) = item,
                  object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) == "message",
                  case let .array(content)? = object["content"] else {
                continue
            }
            for block in content {
                guard case let .object(object) = block,
                      object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) == "output_text",
                      case let .string(text)? = object["text"] else {
                    continue
                }
                parts.append(text)
            }
        }
        return parts.isEmpty ? nil : parts.joined()
    }

    private static func encodedJSONString(for content: GeneratedContent) throws -> String {
        let data = try JSONEncoder.deterministic.encode(content)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func processEvent(
        _ payload: String,
        accumulatedText: inout String,
        latestOutput: inout [JSONValue]?,
        latestOutputText: inout String?,
        toolCallsByID: inout [String: ProviderToolCall]
    ) throws {
        guard payload != "[DONE]", !payload.isEmpty else { return }
        guard let data = payload.data(using: .utf8) else { return }

        let value: JSONValue
        do {
            value = try JSONDecoder().decode(JSONValue.self, from: data)
        } catch {
            return
        }

        if case let .object(object) = value {
            if let type = object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }),
               type == "response.output_text.delta",
               let delta = object["delta"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                accumulatedText += delta
            }

            if let outputText = object["output_text"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                latestOutputText = outputText
            }

            if let response = object["response"], case let .object(responseObject) = response {
                if case let .array(output)? = responseObject["output"] {
                    latestOutput = output
                }
                if let outputText = responseObject["output_text"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                    latestOutputText = outputText
                }
            } else if case let .array(output)? = object["output"] {
                latestOutput = output
            }
        }

        var collected: [ProviderToolCall] = []
        collectToolCalls(from: value, into: &collected)
        for call in collected {
            toolCallsByID[call.id] = call
        }
    }

    /// Pull `reasoning` items out of the final response output array, in
    /// their original order. These are passed back verbatim in the next
    /// request's `input` so the model retains its chain of thought when
    /// `store: false`.
    private static func extractReasoningItems(from output: [JSONValue]?) -> [JSONValue] {
        guard let output else { return [] }
        return output.compactMap { item in
            guard case let .object(object) = item,
                  case let .string(type)? = object["type"],
                  type == "reasoning" else {
                return nil
            }
            return item
        }
    }

    private func send(
        inputs: [JSONValue],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        options: GenerationOptions,
        custom: CustomGenerationOptions
    ) async throws -> CodexStreamingResponse {
        let token = try await tokenProvider()
        let url = resolveCodexURL(from: baseURL)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = defaultLLMRequestTimeout
        request.setValue("Bearer \(token.accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue(token.accountID, forHTTPHeaderField: "chatgpt-account-id")
        request.setValue(originator, forHTTPHeaderField: "originator")
        request.setValue("responses=experimental", forHTTPHeaderField: "OpenAI-Beta")
        request.setValue("text/event-stream", forHTTPHeaderField: "accept")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(sessionID, forHTTPHeaderField: "session_id")
        request.setValue(sessionID, forHTTPHeaderField: "x-client-request-id")
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        // Body keys are already snake_case literals; use the deterministic
        // encoder so nested JSON Schema keys (e.g. `additionalProperties`)
        // are preserved verbatim and dictionary key order is stable.
        let requestBody = try JSONEncoder.deterministic.encode(
            Self.makeRequestBody(
                model: model,
                instructions: resolvedInstructions(instructions),
                inputs: inputs,
                tools: tools,
                promptCacheKey: sessionID,
                options: options,
                custom: custom
            )
        )
        request.httpBody = requestBody

        do {
            return try await withNetworkRetry {
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw CodexLanguageModelError.invalidResponse
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    let data = try await collect(bytes)
                    let message = String(decoding: data, as: UTF8.self)
                    if isRetryableHTTPStatus(httpResponse.statusCode) {
                        throw RetryableServerError(statusCode: httpResponse.statusCode, message: message)
                    }
                    throw CodexLanguageModelError.requestFailed(statusCode: httpResponse.statusCode, message: message)
                }

                return try await parseSSE(bytes: bytes)
            }
        } catch let retryable as RetryableServerError {
            throw CodexLanguageModelError.requestFailed(statusCode: retryable.statusCode, message: retryable.message)
        }
    }

    private func resolvedInstructions(_ instructions: String?) -> String {
        guard let instructions, !instructions.isEmpty else {
            return "You are a helpful assistant."
        }
        return instructions
    }

    private func resolveCodexURL(from baseURL: URL) -> URL {
        let trimmed = baseURL.absoluteString.replacingOccurrences(of: #"/+$"#, with: "", options: .regularExpression)
        if trimmed.hasSuffix("/codex/responses") {
            return URL(string: trimmed)!
        }
        if trimmed.hasSuffix("/codex") {
            return URL(string: trimmed + "/responses")!
        }
        return URL(string: trimmed + "/codex/responses")!
    }

    private func buildInputs(from transcript: Transcript) async throws -> [JSONValue] {
        var input: [JSONValue] = []

        for entry in transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                input.append(
                    .object([
                        "type": .string("message"),
                        "role": .string("user"),
                        "content": .array(Self.convertPromptSegments(prompt.segments, assistant: false))
                    ])
                )
            case let .response(response):
                input.append(
                    .object([
                        "type": .string("message"),
                        "role": .string("assistant"),
                        "content": .array(Self.convertPromptSegments(response.segments, assistant: true))
                    ])
                )
            case let .toolCalls(toolCalls):
                for call in toolCalls {
                    let arguments = try Self.encodedJSONString(for: call.arguments)
                    let callID = providerCallID(fromTranscriptID: call.id)
                    let restoredItemID = providerItemID(fromTranscriptID: call.id)
                    let rememberedItemID = restoredItemID == nil ? await state.itemID(for: callID) : nil
                    var item: [String: JSONValue] = [
                        "type": .string("function_call"),
                        "call_id": .string(callID),
                        "name": .string(call.toolName),
                        "arguments": .string(arguments)
                    ]
                    if let itemID = restoredItemID ?? rememberedItemID {
                        item["id"] = .string(itemID)
                    }
                    input.append(.object(item))
                }
            case let .toolOutput(output):
                input.append(Self.makeFunctionCallOutput(for: output))
            }
        }

        return input
    }

    private func parseSSE(bytes: URLSession.AsyncBytes) async throws -> CodexStreamingResponse {
        var accumulatedText = ""
        var latestOutput: [JSONValue]?
        var latestOutputText: String?
        var toolCallsByID: [String: ProviderToolCall] = [:]

        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            try Self.processEvent(
                payload,
                accumulatedText: &accumulatedText,
                latestOutput: &latestOutput,
                latestOutputText: &latestOutputText,
                toolCallsByID: &toolCallsByID
            )
        }

        let toolCalls = Array(toolCallsByID.values).sorted { $0.id < $1.id }
        let outputText = accumulatedText.isEmpty ? latestOutputText : accumulatedText
        let reasoningItems = Self.extractReasoningItems(from: latestOutput)
        return CodexStreamingResponse(
            output: latestOutput,
            outputText: outputText,
            toolCalls: toolCalls,
            reasoningItems: reasoningItems
        )
    }

    private func collect(_ bytes: URLSession.AsyncBytes) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
        }
        return data
    }
}

// MARK: - CodexSessionState

private actor CodexSessionState {
    // MARK: Internal

    func remember(_ calls: [ProviderToolCall]) {
        for call in calls {
            if let itemID = call.itemID {
                itemIDsByCallID[call.id] = itemID
            }
        }
    }

    func itemID(for callID: String) -> String? {
        itemIDsByCallID[callID]
    }

    // MARK: Private

    private var itemIDsByCallID: [String: String] = [:]
}

// MARK: - OpenResponsesTool

private struct OpenResponsesTool {
    let type: String = "function"
    let name: String
    let description: String
    let parameters: JSONValue?

    var jsonValue: JSONValue {
        var object: [String: JSONValue] = [
            "type": .string(type),
            "name": .string(name),
            "description": .string(description)
        ]
        if let parameters {
            object["parameters"] = parameters
        }
        return .object(object)
    }
}

// MARK: - CodexStreamingResponse

private struct CodexStreamingResponse {
    let output: [JSONValue]?
    let outputText: String?
    let toolCalls: [ProviderToolCall]
    /// Encrypted `reasoning` items emitted by the model in `output`,
    /// preserved verbatim so the caller can replay them in subsequent
    /// requests within the same `LanguageModelSession`.
    let reasoningItems: [JSONValue]
}

// MARK: - CodexLanguageModelError

/// Errors thrown by `CodexLanguageModel` while talking to the Codex backend.
public enum CodexLanguageModelError: LocalizedError, Sendable {
    case invalidResponse
    case requestFailed(statusCode: Int, message: String)
    case unsupportedContentType
    case noResponseGenerated

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "Codex returned an invalid response."
        case let .requestFailed(statusCode, message):
            "Codex request failed with status \(statusCode): \(message)"
        case .unsupportedContentType:
            "CodexLanguageModel only supports text responses."
        case .noResponseGenerated:
            "Codex did not produce any text or tool calls."
        }
    }
}
