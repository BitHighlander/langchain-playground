import * as dotenv from "dotenv";
dotenv.config();
//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI } from "langchain/llms/openai";
//Import the Vector DB QA chain
import { VectorDBQAChain } from "langchain/chains";
//Import the Hierarchical Navigable Small World Graphs vector store (you'll learn
//how it is used later in the code)
import { HNSWLib } from "langchain/vectorstores";
//Import OpenAI embeddings (you'll learn
//how it is used later in the code)
import { OpenAIEmbeddings } from "langchain/embeddings";
//Import the text splitter (you'll learn
//how it is used later in the code)
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
//Import file stystem node module
import * as fs from "fs";
export const run = async () => {
    //Instantiante the OpenAI model
    //Pass the "temperature" parameter which controls the RANDOMNESS of the model's output. A lower temperature will result in more predictable output, while a higher temperature will result in more random output. The temperature parameter is set between 0 and 1, with 0 being the most predictable and 1 being the most random
    const model = new OpenAI({ temperature: 0.9 });
    //memory
    // let files = [
    //     'test',
    //     'caips',
    //     'bitcoinbook',
    //     'desktop'
    // ]
    let files = fs.readdirSync("./data/");
    console.log("files: ", files);
    let ALL_MEMORY = [];
    for (let i = 0; i < files.length; i++) {
        console.log("filename: ", files[i]);
        //Load in the file containing the content on which we will be performing Q&A
        //The answers to the questions are contained in this file
        const text = fs.readFileSync("./data/" + files[i], "utf8");
        ALL_MEMORY.push(text);
    }
    //console.log("text: ",text)
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000 });
    //Create documents from the split text as required by subsequent calls
    const docs = await textSplitter.createDocuments(ALL_MEMORY);
    //Calls out to the model's (OpenAI's) endpoint passing the prompt. This call returns a string
    // const res = await model.call(
    //     "What would be a good company name a company that makes colorful socks?"
    // );
    // console.log({ res });
    //Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
    //Github repo
    // const loader = new GithubRepoLoader(
    //     // "https://github.com/BitHighlander/langchain-playground",
    //     "https://github.com/hwchase17/langchainjs",
    //     { branch: "main", recursive: true, unknown: "warn" }
    // );
    // const repo = await loader.load();
    // console.log({ docs });
    //
    // const docs = await textSplitter.splitDocuments(repo);
    //Create the vector store from OpenAIEmbeddings
    //OpenAIEmbeddings is used to create a vector representation of a text in the documents.
    //Converting the docs to the vector format and storing it in the vectorStore enables LangChain.js
    //to perform similarity searches on the "await chain.call"
    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    //Create the LangChain.js chain consting of the LLM and the vector store
    const chain = VectorDBQAChain.fromLLM(model, vectorStore);
    //Ask the question that will use content from the file to find the answer
    //The way this all comes together is that when the following call is made, the "query" (question) is used to
    //search the vector store to find chunks of text that is similar to the text in the "query" (question). Those
    //chunks of text are then sent to the LLM to find the answer to the "query" (question). This is done because,
    //as explained earlier, the LLMs have a limit in size of the text that can be sent to them
    const res = await chain.call({
        input_documents: docs,
        query: "tell me able the rest api? can pioneer give me the chainId and accountNumber from an osmosis address or pubkey?",
    });
    console.log({ res });
    //
    //Instantiate the BufferMemory passing the memory key for storing state
    // const memory = new BufferMemory({ memoryKey: "chat_history" });
    //Create the template. The template is actually a "parameterized prompt". A "parameterized prompt" is a prompt in which the input parameter names are used and the parameter values are supplied from external input
    //Note the input variables {chat_history} and {input}
    // const template = `The following is a conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    //   Current conversation:
    //   {chat_history}
    //   Human: {input}
    //   AI:`;
    //
    // //Instantiate "PromptTemplate" passing the prompt template string initialized above
    // const prompt = PromptTemplate.fromTemplate(template);
    //Instantiate LLMChain, which consists of a PromptTemplate, an LLM and memory.
    // const chain = new LLMChain({ llm: model, prompt, memory });
    // //Run the chain passing a value for the {input} variable. The result will be stored in {chat_history}
    // const res1 = await chain.call({ input: "Hi! I'm Morpheus." });
    // console.log({ res1 });
    //
    // //Run the chain again passing a value for the {input} variable. This time, the response from the last run ie. the  value in {chat_history} will alo be passed as part of the prompt
    // const res2 = await chain.call({ input: "What's my name?" });
    // console.log({ res2 });
    //
    // //BONUS!!
    // const res3 = await chain.call({ input: "Which epic movie was I in and who was my protege?" });
    // console.log({ res3 });
    //search
    // const tools = [docs];
    //
    // //Construct an agent from an LLM and a list of tools
    // //"zero-shot-react-description" tells the agent to use the ReAct framework to determine which tool to use. The ReAct framework determines which tool to use based solely on the tool’s description. Any number of tools can be provided. This agent requires that a description is provided for each tool.
    // // const executor = await initializeAgentExecutor(
    // //     tools,
    // //     model,
    // //     "zero-shot-react-description"
    // // );
    // // console.log("Loaded agent.");
};
run();
