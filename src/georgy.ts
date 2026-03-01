import { ChatOpenAI } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";

const qdrantUrl = import.meta.env.QDRANT_URL;

const apiKey = import.meta.env.API_KEY_OPENROUTER_OPENAI;
const baseURL = "https://openrouter.ai/api/v1";
const model = new ChatOpenAI({
  model: "gpt-4.1-nano",
  configuration: {
    apiKey,
    baseURL,
  },
});
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: qdrantUrl,
  collectionName: "test",
});
