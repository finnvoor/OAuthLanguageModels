// The `LanguageModel` / `LanguageModelExecutor` API is new in the 27-series
// SDKs (FoundationModels module version 2.0+). Gating on the module version
// keeps this file out of older SDKs (e.g. Xcode 26) where those symbols don't
// exist; on those SDKs only the `AnyLanguageModel` conformance is built.
#if canImport(FoundationModels, _version: 2.0)
import Foundation
import FoundationModels

// MARK: - FoundationModels.LanguageModel conformance

@available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
@available(tvOS, unavailable) extension AnthropicOAuthLanguageModel: FoundationModels.LanguageModel {
    public var capabilities: LanguageModelCapabilities {
        .init([.toolCalling, .vision])
    }

    public var executorConfiguration: Executor.Configuration {
        .init()
    }

    public struct Executor: LanguageModelExecutor {
        // MARK: Lifecycle

        public init(configuration _: Configuration) throws {}

        // MARK: Public

        public struct Configuration: Hashable, Sendable {
            public init() {}
        }

        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: AnthropicOAuthLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let tools = try request.enabledToolDefinitions.map {
                try makeAnthropicTool(name: $0.name, description: $0.description, schema: $0.parameters)
            }

            let payload = try await model.send(
                messages: Self.buildMessages(from: request.transcript),
                instructions: Self.instructions(from: request.transcript),
                tools: tools.isEmpty ? nil : tools,
                parameters: AnthropicRequestParameters(
                    temperature: request.generationOptions.temperature,
                    maxTokens: request.generationOptions.maximumResponseTokens
                )
            )

            var emittedAnything = false
            for block in payload.content {
                switch block {
                case let .text(text):
                    emittedAnything = true
                    await channel.send(.response(action: .appendText(text.text, tokenCount: 0)))
                case let .toolUse(use):
                    emittedAnything = true
                    await channel.send(
                        .toolCalls(
                            action: .toolCall(
                                id: use.id,
                                name: use.name,
                                action: .appendArguments(use.argumentsJSONString, tokenCount: 0)
                            )
                        )
                    )
                default:
                    break
                }
            }

            if !emittedAnything {
                await channel.send(.response(action: .appendText("", tokenCount: 0)))
            }
        }

        // MARK: Private

        private static func instructions(from transcript: Transcript) -> String? {
            let text = transcript.compactMap { entry -> String? in
                guard case let .instructions(instructions) = entry else { return nil }
                return plainText(from: instructions.segments)
            }.joined(separator: "\n")
            return text.isEmpty ? nil : text
        }

        private static func buildMessages(from transcript: Transcript) -> [AnthropicRequest.Message] {
            var messages: [AnthropicRequest.Message] = []
            for entry in transcript {
                switch entry {
                case .instructions, .reasoning:
                    break
                case let .prompt(prompt):
                    messages.append(.init(role: "user", content: convertSegments(prompt.segments)))
                case let .response(response):
                    messages.append(.init(role: "assistant", content: convertSegments(response.segments)))
                case let .toolCalls(toolCalls):
                    let blocks = toolCalls.map { call in
                        makeAnthropicToolUseBlock(
                            id: call.id,
                            name: call.toolName,
                            argumentsJSONString: call.arguments.jsonString
                        )
                    }
                    messages.append(.init(role: "assistant", content: blocks))
                case let .toolOutput(toolOutput):
                    messages.append(
                        .init(
                            role: "user",
                            content: [
                                .toolResult(.init(toolUseID: toolOutput.id, content: convertSegments(toolOutput.segments)))
                            ]
                        )
                    )
                @unknown default:
                    break
                }
            }
            return messages
        }

        private static func convertSegments(_ segments: [Transcript.Segment]) -> [AnthropicResponse.ContentBlock] {
            segments.compactMap { segment -> AnthropicResponse.ContentBlock? in
                switch segment {
                case let .text(text):
                    return .text(.init(text: text.content))
                case let .structure(structured):
                    switch structured.content.kind {
                    case let .string(string):
                        return .text(.init(text: string))
                    default:
                        return .text(.init(text: structured.content.jsonString))
                    }
                case let .attachment(attachment):
                    if case let .image(image) = attachment.content, let url = image.url {
                        return .image(.init(url: url.absoluteString))
                    }
                    return nil
                @unknown default:
                    return nil
                }
            }
        }

        private static func plainText(from segments: [Transcript.Segment]) -> String {
            segments.compactMap { segment -> String? in
                switch segment {
                case let .text(text): text.content
                case let .structure(structured): structured.content.jsonString
                default: nil
                }
            }.joined()
        }
    }
}
#endif
