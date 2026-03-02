import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";

const qdrantUrl = import.meta.env.QDRANT_URL;
const qdrantApiKey = import.meta.env.API_KEY_QDRANT;
const unstructuredApiKey = import.meta.env.API_KEY_UNSTRUCTURED;
const unstructuredApiUrl = "https://platform.unstructuredapp.io/api/v1";
import.meta.env.UNSTRUCTURED_API_URL || "https://api.unstructuredapp.io";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

interface UnstructuredElement {
  type: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface UnstructuredResponse {
  elements: UnstructuredElement[];
}

/**
 * Unstructure a document using the Unstructured API
 */
async function unstructureDocument(
  fileContent: string,
  fileType: string,
): Promise<UnstructuredElement[]> {
  const formData = new FormData();
  const blob = new Blob([fileContent], { type: "application/octet-stream" });
  formData.append("files", blob, `document.${fileType}`);

  const response = await fetch(`${unstructuredApiUrl}/general/v0/general`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${unstructuredApiKey}`,
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Unstructured API error: ${response.statusText}`);
  }

  const data: UnstructuredResponse = await response.json();
  return data.elements;
}

/**
 * Convert unstructured elements to LangChain documents
 */
function elementsToDocuments(
  elements: UnstructuredElement[],
  sourceMetadata?: Record<string, unknown>,
): Document[] {
  return elements
    .filter((el) => el.text && el.text.trim().length > 0)
    .map((el, index) => {
      return new Document({
        pageContent: el.text,
        metadata: {
          type: el.type,
          index: index,
          ...el.metadata,
          ...sourceMetadata,
        },
      });
    });
}

/**
 * Process documents: unstructure them and add to Qdrant vector store
 */
export async function processDocuments(
  documents: Array<{ content: string; fileType: string; id?: string }>,
  collectionName: string = "documents",
): Promise<void> {
  if (!unstructuredApiKey) {
    throw new Error("UNSTRUCTURED_API_KEY environment variable is not set");
  }

  if (!qdrantUrl) {
    throw new Error("QDRANT_URL environment variable is not set");
  }

  try {
    // Get or create vector store
    let vectorStore: QdrantVectorStore;

    try {
      vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
        url: qdrantUrl,
        apiKey: qdrantApiKey,
        collectionName,
      });
    } catch {
      // Collection doesn't exist, will be created when adding documents
      vectorStore = new QdrantVectorStore(embeddings, {
        url: qdrantUrl,
        apiKey: qdrantApiKey,
        collectionName,
      });
    }

    // Process each document
    for (const doc of documents) {
      console.log(`Processing document: ${doc.id || "unnamed"}`);

      // Unstructure the document
      const elements = await unstructureDocument(doc.content, doc.fileType);
      console.log(`Extracted ${elements.length} elements from document`);

      // Convert to LangChain documents
      const langchainDocs = elementsToDocuments(elements, {
        documentId: doc.id || Date.now().toString(),
      });

      // Add to vector store
      if (langchainDocs.length > 0) {
        await vectorStore.addDocuments(langchainDocs);
        console.log(
          `Added ${langchainDocs.length} documents to Qdrant collection "${collectionName}"`,
        );
      }
    }

    console.log("Document processing completed successfully");
  } catch (error) {
    console.error("Error processing documents:", error);
    throw error;
  }
}

/**
 * Search for documents in the vector store
 */
export async function searchDocuments(
  query: string,
  collectionName: string = "documents",
  limit: number = 5,
): Promise<Document[]> {
  if (!qdrantUrl) {
    throw new Error("QDRANT_URL environment variable is not set");
  }

  try {
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: qdrantUrl,
        apiKey: qdrantApiKey,
        collectionName,
      },
    );

    const results = await vectorStore.similaritySearch(query, limit);
    return results;
  } catch (error) {
    console.error("Error searching documents:", error);
    throw error;
  }
}
