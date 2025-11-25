import "dotenv/config";

import { ChatGroq } from "@langchain/groq";
import { createAgent, humanInTheLoopMiddleware, tool } from "langchain";
import { z } from "zod";
import readline from "node:readline/promises";
import { connectVectorStore } from "./prepare.js"; 
import { MemorySaver, Command } from "@langchain/langgraph";


// --- Initialization ---
const vectorStore = await connectVectorStore();

// Initializing the MemorySaver Checkpointer
const memorySaver = new MemorySaver();


// 1. Rag Search tool
const ragSearch = tool(
  async ({ query }) => {
    console.log("RAG tool calling...");
    const results = await vectorStore.similaritySearch(query, 5);
    if (!results.length) return "No relevant emails found.";
    return results
      .map((doc, i) => `Email ${i + 1}:\n${doc.pageContent}`)
      .join("\n\n");
  },
  {
    name: "ragQuery",
    description: "Search inbox emails using vector SIM.",
    schema: z.object({ query: z.string() }),
  }
);


// 2. SEND Email Tool (Triggers the human interrupt)
const sendEmailTool = tool(
  async ({ recipient, subject, body }) => {
    console.log(`\n✅ Email Approved and SENT to: ${recipient}`);
    console.log("Subject:", subject);
    console.log("Body:\n", body);
  
    return `Email successfully sent to ${recipient}!`;
  },
  {
    name: "sendEmailTool",
    description: "Sends the final approved email to a recipient.",
    schema: z.object({
      recipient: z.string().describe("The recipient's email address."),
      subject: z.string().describe("The final email subject line."),
      body: z.string().describe("The complete, formatted email body content."),
    }),
  }
);


// --- LLM & Agent Configuration ---
const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  systemPrompt: `
    You are an Email Assistant. Your primary goal is to propose an email draft immediately via the 'sendEmailTool'.
    1. **MANDATORY**: If the user asks you to draft/send an email and provides the recipient, you MUST **compose the complete email body** based on their request and call 'sendEmailTool' immediately. DO NOT ASK FOR MORE DETAILS. Use generic details if specific ones are missing (e.g., use 'Refund Request' as subject if user only says 'refund email').
    2. Use the 'ragQuery' tool first if the user's request involves previous emails.
    3. Do not output any conversational text before calling a tool.
  `,
});

const agent = createAgent({
  model: llm,
  tools: [ragSearch, sendEmailTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: { sendEmailTool: true },
      descriptionPrefix: "Send Email Approval",
      // Checkpointer remains missing from middleware as requested
    }),
  ],
  checkpointer: new MemorySaver(), // Checkpointer remains at root as requested
});


// FIX: Removed unnecessary global 'pendingArgs = null;'


async function main() {
    console.log("Email Agent Ready. Ask me to 'draft an email...'");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    let interrupts = [];

    while (true) {
        const query = await rl.question("You: ");

        if (query === "/bye") break;

        let response;

    
        if (interrupts.length) {
            const decision = query === "1" ? "approve" : "reject";

            const command = new Command({
                resume: {
                    [interrupts[0].id]: {
                        decisions: [{ type: decision }],
                    },
                },
            });

            response = await agent.invoke(command, {
                configurable: { thread_id: "1" },
            });
        }

       
        else {
            response = await agent.invoke(
                { messages: [{ role: "user", content: query }] },
                { configurable: { thread_id: "1" } }
            );
        }

        interrupts = [];
        let output = "";

       
        if (response?.__interrupt__?.length) {
    console.log(response.__interrupt__)
            const interrupt = response.__interrupt__[0];
            interrupts.push(interrupt);

            const req = interrupt.value.actionRequests?.[0];
            console.log(req)
           
            const args = req?.args ||  {}; 
            
           

            output += `\n*** ⏸️ HUMAN REVIEW REQUIRED (Enter '1' to Approve) ***\n`;
            output += `Tool Requested: ${req?.name || 'N/A'}\n`;
            output += "--- Proposed Email Draft ---\n";
            
            
            output += `Recipient: ${args.recipient || 'N/A'}\n`;
            output += `Subject: ${args.subject || 'N/A'}\n\n`;
            output += `Body:\n${(args.body || 'Email body not generated.').trim()}\n`;
         

            output += "Choose:\n1. approve\n2. reject\n";
            output += "(Enter '1' to approve and send, any other key to reject)";
        }

       
        else {
            output += response.messages[response.messages.length - 1].content;
        }

        console.log(output);
    }

    rl.close();
}

main();