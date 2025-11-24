import { ChatGroq } from "@langchain/groq"
import {tool , CreateAgent} from 'langchain'
const getEmails = tool(()=>{

},{
    name:"get_emails",
    description:""
})
const ragQuery = tool(()=>{

},{
 name:"ragQuery",
 description:""
})
const draftEmailtool = tool(()=>{

},{

})

const agent = CreateAgent({
    model:llm,
    

})
const llm = new ChatGroq({
    model:'openai/gpt-oss-120b',
    temperature:0,
    maxConcurrency:0,
})