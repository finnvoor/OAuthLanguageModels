import AnyLanguageModel
import Foundation

// MARK: - ProviderToolCall

struct ProviderToolCall {
    let id: String
    let itemID: String?
    let name: String
    let arguments: GeneratedContent
}

// MARK: - ProviderToolInvocationResult

struct ProviderToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

// MARK: - ProviderToolResolutionOutcome

enum ProviderToolResolutionOutcome {
    case stop(calls: [Transcript.ToolCall])
    case invocations([ProviderToolInvocationResult])
}

func resolveToolCalls(
    _ providerCalls: [ProviderToolCall],
    session: LanguageModelSession
) async throws -> ProviderToolResolutionOutcome {
    if providerCalls.isEmpty { return .invocations([]) }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools where toolsByName[tool.name] == nil {
        toolsByName[tool.name] = tool
    }

    let transcriptCalls = providerCalls.map {
        Transcript.ToolCall(id: toolCallTranscriptID(callID: $0.id, itemID: $0.itemID), toolName: $0.name, arguments: $0.arguments)
    }

    if let delegate = session.toolExecutionDelegate {
        await delegate.didGenerateToolCalls(transcriptCalls, in: session)
    }

    guard !transcriptCalls.isEmpty else { return .invocations([]) }

    let decisions: [ToolExecutionDecision]
    if let delegate = session.toolExecutionDelegate {
        var collected: [ToolExecutionDecision] = []
        collected.reserveCapacity(transcriptCalls.count)
        for call in transcriptCalls {
            let decision = await delegate.toolCallDecision(for: call, in: session)
            if case .stop = decision {
                return .stop(calls: transcriptCalls)
            }
            collected.append(decision)
        }
        decisions = collected
    } else {
        decisions = Array(repeating: .execute, count: transcriptCalls.count)
    }

    var results: [ProviderToolInvocationResult] = []
    results.reserveCapacity(transcriptCalls.count)

    for (index, call) in transcriptCalls.enumerated() {
        switch decisions[index] {
        case .stop:
            return .stop(calls: transcriptCalls)
        case let .provideOutput(segments):
            let output = Transcript.ToolOutput(id: call.id, toolName: call.toolName, segments: segments)
            if let delegate = session.toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: session)
            }
            results.append(.init(call: call, output: output))
        case .execute:
            guard let tool = toolsByName[call.toolName] else {
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: [.text(.init(content: "Tool not found: \(call.toolName)"))]
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(.init(call: call, output: output))
                continue
            }

            do {
                let output = try await callTool(tool, arguments: call.arguments, callID: call.id)
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(.init(call: call, output: output))
            } catch {
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didFailToolCall(call, error: error, in: session)
                }
                throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
            }
        }
    }

    return .invocations(results)
}

func toolCallTranscriptID(callID: String, itemID: String?) -> String {
    guard let itemID, !itemID.isEmpty else { return callID }
    return "codex-response-item:\(itemID):\(callID)"
}

func providerCallID(fromTranscriptID id: String) -> String {
    guard id.hasPrefix("codex-response-item:") else { return id }
    return String(id.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false).last ?? Substring(id))
}

func providerItemID(fromTranscriptID id: String) -> String? {
    guard id.hasPrefix("codex-response-item:") else { return nil }
    let parts = id.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
    guard parts.count == 3, !parts[1].isEmpty else { return nil }
    return String(parts[1])
}

private func callTool<T: Tool>(
    _ tool: T,
    arguments: GeneratedContent,
    callID: String
) async throws -> Transcript.ToolOutput {
    let parsedArguments = try T.Arguments(arguments)
    let output = try await tool.call(arguments: parsedArguments)

    let outputSegments: [Transcript.Segment] = if let structured = output as? any ConvertibleToGeneratedContent {
        [.structure(.init(source: tool.name, content: structured.generatedContent))]
    } else if let stringOutput = output as? String {
        [.text(.init(content: stringOutput))]
    } else {
        [.text(.init(content: output.promptRepresentation.description))]
    }

    return Transcript.ToolOutput(id: callID, toolName: tool.name, segments: outputSegments)
}

func emptyResponseContent<Content: Generable>(
    for type: Content.Type
) throws -> (content: Content, rawContent: GeneratedContent) {
    if type == String.self {
        let raw = GeneratedContent("")
        return ("" as! Content, raw)
    }

    let emptyObject = GeneratedContent(properties: [:])
    if let content = try? Content(emptyObject) {
        return (content, emptyObject)
    }

    let nullContent = GeneratedContent(kind: .null)
    if let content = try? Content(nullContent) {
        return (content, nullContent)
    }

    throw GeneratedContentConversionError.typeMismatch
}
