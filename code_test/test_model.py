from Model.gemini import GeminiClient

client = GeminiClient(use_rag=False)

response = client.chat("21% 파티가 뭔가요?")

print(response)