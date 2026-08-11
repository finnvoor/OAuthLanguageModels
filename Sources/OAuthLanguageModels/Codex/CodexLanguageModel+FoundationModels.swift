// The `LanguageModel` / `LanguageModelExecutor` API is new in the 27-series
// SDKs (FoundationModels module version 2.0+). Gating on the module version
// keeps this file out of older SDKs (e.g. Xcode 26) where those symbols don't
// exist; on those SDKs only the `AnyLanguageModel` conformance is built.
#if canImport(FoundationModels, _version: 2.0)
import Foundation
import FoundationModels

// MARK: - FoundationModels.LanguageModel conformance

@available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
@available(tvOS, unavailable) extension CodexLanguageModel: FoundationModels.LanguageModel {
    public var capabilities: LanguageModelCapabilities {
        .init([.toolCalling, .reasoning])
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
            model: CodexLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let tools = request.enabledToolDefinitions.map {
                makeOpenResponsesTool(name: $0.name, description: $0.description, schema: $0.parameters)
            }

            let response = try await model.send(
                inputs: Self.buildInputs(from: request.transcript),
                instructions: Self.instructions(from: request.transcript),
                tools: tools.isEmpty ? nil : tools,
                parameters: CodexRequestParameters(
                    temperature: request.generationOptions.temperature,
                    maxOutputTokens: request.generationOptions.maximumResponseTokens
                )
            )

            for call in response.toolCalls {
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: call.id,
                            name: call.name,
                            action: .appendArguments(call.argumentsJSON, tokenCount: 0)
                        )
                    )
                )
            }

            if let text = response.text, !text.isEmpty {
                await channel.send(.response(action: .appendText(text, tokenCount: 0)))
            } else if response.toolCalls.isEmpty {
                await channel.send(.response(action: .appendText("", tokenCount: 0)))
            }
        }

        // MARK: Private

        private static func instructions(from transcript: Transcript) -> String? {
            let text = transcript.compactMap { entry -> String? in
                guard case let .instructions(instructions) = entry else { return nil }
                return joined(instructions.segments)
            }.joined(separator: "\n")
            return text.isEmpty ? nil : text
        }

        private static func buildInputs(from transcript: Transcript) -> [CodexInputItem] {
            var input: [CodexInputItem] = []
            for entry in transcript {
                switch entry {
                case .instructions, .reasoning:
                    break
                case let .prompt(prompt):
                    input.append(.userMessage(textSegments: texts(prompt.segments), imageURLs: imageURLs(prompt.segments)))
                case let .response(response):
                    input.append(.assistantMessage(textSegments: texts(response.segments)))
                case let .toolCalls(toolCalls):
                    for call in toolCalls {
                        input.append(
                            .functionCall(
                                itemID: call.id,
                                callID: call.id,
                                name: call.toolName,
                                argumentsJSON: call.arguments.jsonString
                            )
                        )
                    }
                case let .toolOutput(output):
                    input.append(.functionCallOutput(callID: output.id, output: joined(output.segments)))
                @unknown default:
                    break
                }
            }
            return input
        }

        private static func texts(_ segments: [Transcript.Segment]) -> [String] {
            segments.compactMap { segment -> String? in
                switch segment {
                case let .text(text): text.content
                case let .structure(structured): structured.content.jsonString
                default: nil
                }
            }
        }

        private static func imageURLs(_ segments: [Transcript.Segment]) -> [String] {
            segments.compactMap { segment -> String? in
                guard case let .attachment(attachment) = segment,
                      case let .image(image) = attachment.content,
                      let url = image.url else {
                    return nil
                }
                return url.absoluteString
            }
        }

        private static func joined(_ segments: [Transcript.Segment]) -> String {
            texts(segments).joined(separator: "\n")
        }
    }
}
#endif
