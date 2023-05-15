//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI } from "langchain/llms/openai";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
//Import the Chains module
import { LLMChain } from "langchain/chains";
//Import the PromptTemplate module
import { PromptTemplate } from "langchain/prompts";
//Import the BufferMemory module
import { BufferMemory } from "langchain/memory";
//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv";
dotenv.config();
export const run = async () => {
    //Instantiante the OpenAI model
    //Pass the "temperature" parameter which controls the RANDOMNESS of the model's output. A lower temperature will result in more predictable output, while a higher temperature will result in more random output. The temperature parameter is set between 0 and 1, with 0 being the most predictable and 1 being the most random
    const model = new OpenAI({ temperature: 0.9 });
    //Calls out to the model's (OpenAI's) endpoint passing the prompt. This call returns a string
    // const res = await model.call(
    //     "What would be a good company name a company that makes colorful socks?"
    // );
    // console.log({ res });
    //Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
    const loader = new GithubRepoLoader("https://github.com/hwchase17/langchainjs", { branch: "main", recursive: false, unknown: "warn" });
    const docs = await loader.load();
    console.log({ docs });
    //Instantiate the BufferMemory passing the memory key for storing state
    const memory = new BufferMemory({ memoryKey: "chat_history" });
    //Create the template. The template is actually a "parameterized prompt". A "parameterized prompt" is a prompt in which the input parameter names are used and the parameter values are supplied from external input
    //Note the input variables {chat_history} and {input}
    const template = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
      Current conversation:
      {chat_history}
      Human: {input}
      AI:`;
    //Instantiate "PromptTemplate" passing the prompt template string initialized above
    const prompt = PromptTemplate.fromTemplate(template);
    //Instantiate LLMChain, which consists of a PromptTemplate, an LLM and memory.
    const chain = new LLMChain({ llm: model, prompt, memory });
    //Run the chain passing a value for the {input} variable. The result will be stored in {chat_history}
    const res1 = await chain.call({ input: "Hi! I'm Morpheus." });
    console.log({ res1 });
    //Run the chain again passing a value for the {input} variable. This time, the response from the last run ie. the  value in {chat_history} will alo be passed as part of the prompt
    const res2 = await chain.call({ input: "What's my name?" });
    console.log({ res2 });
    //BONUS!!
    const res3 = await chain.call({ input: "Which epic movie was I in and who was my protege?" });
    console.log({ res3 });
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
