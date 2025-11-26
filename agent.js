import "dotenv/config";

import { ChatGroq } from "@langchain/groq";
import { createAgent, humanInTheLoopMiddleware, tool } from "langchain";
import { z } from "zod";
import readline from "node:readline/promises";
import { connectVectorStore } from "./prepare.js"; 
import { MemorySaver, Command } from "@langchain/langgraph";



const vectorStore = await connectVectorStore();


const memorySaver = new MemorySaver();



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



const sendEmailTool = tool(
async ({ recipient, subject, body }) => {
    
   console.log(`updated send body: ${body}`)
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
      interruptOn: { sendEmailTool: true  },
    skipOnResume: true,
      descriptionPrefix: "Send Email Approval",
      
    }),
  ],
  checkpointer: new MemorySaver(), 
});



async function main() {
    
const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    let interrupts = [];
    while (true) {
        const query = await rl.question("You: ");

        if (query === "bye") break;

        let response;

        
        if (interrupts.length) {
            let command;

            
            if (query === "1") {
                command = new Command({
                    resume: {
                        [interrupts[0].id]: {
                            decisions: [{ type: "approve" }],
                        },
                    },
                });
            }
       
if (query === "2") {
    const req = interrupts[0].value.actionRequests[0];

    console.log("\n--- Edit Email Body ---");
    console.log("Current body:\n" + req.args.body);
    console.log("\nType the full corrected email body below.");
    console.log("Press ENTER on an empty line to finish.\n");

    const editedLines = [];
    while (true) {
        const line = await rl.question("");
        if (!line.trim()) break;
        editedLines.push(line);
    }

    const updatedBody = editedLines.length ? editedLines.join("\n").trim() : req.args.body;


    const finalArgs = {
        recipient: req.args.recipient,
        subject: req.args.subject,
        body: updatedBody, 
    };

    command = new Command({
        resume: {
            [interrupts[0].id]: {
                decisions: [
                    {
                        type: "edit", 
                        editedAction: {
                            name: req.name,
                            args: finalArgs, 
                        }
                    },
                ],
            },
        },
    });
}


            else {
                command = new Command({
                    resume: {
                        [interrupts[0].id]: {
                            decisions: [{ type: "reject" }],
                        },
                    },
                });
            }


            response = await agent.invoke(command, {
                configurable: { thread_id: "1" },
            });
           

            interrupts = [];
            continue;
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
           
            const interrupt = response.__interrupt__[0];
            interrupts.push(interrupt);

            const req = interrupt.value.actionRequests?.[0];
            
            
            const args = req?.args || {}; 
            
            

            output += `\n*** ⏸️ HUMAN REVIEW REQUIRED (Enter '1' to Approve) ***\n`;
            output += `Tool Requested: ${req?.name || 'N/A'}\n`;
            output += "--- Proposed Email Draft ---\n";
            
            
            output += `Recipient: ${args.recipient || 'N/A'}\n`;
            output += `Subject: ${args.subject || 'N/A'}\n\n`;
            output += `Body:\n${(args.body || 'Email body not generated.').trim()}\n`;
            

            output += "Choose:\n1. approve\n2. edit \n3. reject\n";
            output += "(Enter '1' to approve and send, '2' to edit and any other key to reject)";
        }

        
   
        else {
            const lastMessage = response.messages[response.messages.length - 1];

        
          
            if (lastMessage?.type === "tool") {
                
                output += lastMessage.content;
            } 
           
            else { 
                output += lastMessage?.content || "No final content received.";
            }
        }

        console.log(output);
    }

    rl.close();
}

main()

