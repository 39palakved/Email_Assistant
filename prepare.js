// prepare.js
import "dotenv/config";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

import { emails } from "./data/emails.js";


export async function connectVectorStore() {
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const pineconeIndex = pc.Index("email-index");

  const vectorStore = await PineconeStore.fromExistingIndex(
    embeddings,
    {
      pineconeIndex,
      maxRetries: 5,
    }
  );

  console.log(" Vector Store Connected Successfully");
  return vectorStore;
}




export async function IndexEmails() {
  try {
    console.log("Fetching emails...");
    console.log(`Total emails: ${emails.length}`);

    if (!emails.length) {
      console.log("⚠ No emails found to index.");
      return null;
    }

    const vectorStore = await connectVectorStore();

    const docs = emails.map((email, index) => ({
      id: `email-${index}`,
      pageContent: `
        FROM: ${email.from}
        SUBJECT: ${email.subject}
        BODY: ${email.body}
        DATE: ${email.date}
      `,
      metadata: { from: email.from, date: email.date }
    }));

    await vectorStore.addDocuments(docs);
    console.log(" All Emails Indexed Successfully!");


    return vectorStore;

  } catch (error) {
    console.error("Error indexing emails:", error.message);
    return null;
  }
}
