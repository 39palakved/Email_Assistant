import "dotenv/config";

import { ChatGroq } from "@langchain/groq";
import { createAgent, humanInTheLoopMiddleware, tool } from "langchain";
import { z } from "zod";
import readline from "node:readline/promises";
import { connectVectorStore } from "./prepare.js";
import { MemorySaver, Command } from "@langchain/langgraph";

// --- Tools ---
const vectorStore = await connectVectorStore();

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
    description: "Search the inbox for emails related to a query using vector similarity.",
    schema: z.object({ query: z.string() }),
  }
);

const generateDraftTool = tool(
  async ({ recipient, details }) => {
    const llm = new ChatGroq({ model: "openai/gpt-oss-120b", temperature: 0 });
    const response = await llm.invoke({
      messages: [
        {
          role: "user",
          content: `Draft an email to ${recipient} using details: ${details}`,
        },
      ],
    });
    return response.content;
  },
  {
    name: "generateDraftTool",
    description: "Generates an email draft using LLM",
    schema: z.object({
      recipient: z.string(),
      details: z.string(),
    }),
  }
);

const draftEmailTool = tool(
  async ({ subject, body }) => {
    console.log("Draft email tool called...");
    console.log("Subject:", subject);
    console.log("Body:", body);
    return "Draft email saved successfully!";
  },
  {
    name: "draftEmailTool",
    description: "Saves the approved email draft.",
    schema: z.object({ subject: z.string(), body: z.string() }),
  }
);

// --- LLM & Agent ---
const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  systemPrompt: `
    You are an Email Assistant. 
    When the user asks to draft an email, generate the email subject and body.
    Do not send it yet; the draft will be intercepted by middleware for approval.
  `,
});

const agent = createAgent({
  model: llm,
  tools: [ragSearch, generateDraftTool, draftEmailTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: { draftEmailTool: true },
      checkpointer: new MemorySaver(),
    }),
  ],
});


 async function main(){
    const rl = readline.createInterface({input:process.stdin , output:process.stdout})
    let interrupts = []
    
    while(true){
        const query = await rl.question("You: ")
        if(query === '/bye'){
            break;
        }

        const response = await agent.invoke(
            interrupts.length 
            ?
            new Command({resume: {
                [interrupts?.[0]?.id]:{
                    decisions :[{type: query === '1'? 'approve' : 'reject'}]
                }
            }})
    :{
        messages:[
            {role:'user',
              content:query
            }
            
        ]
    },
    {
        configurable:{thread_id: '1'}
    }
);
      interrupts =[];
      let output = ""
    if(response?.__interrupt__?.length){
interrupts.push(response.__interrupt__[0])
output +=  response.__interrupt__[0].value.actionRequests[0].description + "\n\n"
output += "Choose:\n"
output += response.__interrupt__[0].value.reviewConfigs[0].allowedDecisions.filter((decision=> decision !== 'edit')).map((decision,idx)=>
    `${idx +1}. ${decision}`
).join("\n");

}
else{
    output += response.messages[response.messages.length-1].content
}
console.log(output);
    }
    rl.close();
}
main();
 