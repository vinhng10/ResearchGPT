import re
import aiohttp
import json
import pendulum
from word2number import w2n

import uuid
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from FlagEmbedding import FlagReranker
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    ChatPromptTemplate,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


class Document(BaseModel):
    content: str


class Message(BaseModel):
    role: str = "user"
    content: str


class Messages(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.8
    user: str


class Embedding(BaseModel):
    author: str
    chat: str | None = None
    context: str | None = None
    content: str
    boundary: list[str]


class Environment(BaseSettings):
    llm_url: str = "https://development-llm.dbinno.com/v1"
    embed_model: str = "BAAI/bge-large-en-v1.5"
    vector_store_url: str = "https://development-qdrant.dbinno.com"
    temperature: float = 0.1
    similarity_top_k: int = 1
    chunk_size: int = 316
    chunk_overlap: int = 64


env = Environment()
app = FastAPI()

origins = [
    "https://development-app.dbinno.com",
    "https://development-api.dbinno.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Settings.llm = OpenAI(
    api_base=env.llm_url, temperature=env.temperature, max_tokens=1024
)
Settings.embed_model = FastEmbedEmbedding(
    model_name=env.embed_model, cache_dir="/storage"
)

# Sync Qdrant client:
client = QdrantClient(url=env.vector_store_url, port=443)
vector_store = QdrantVectorStore(client=client, collection_name="Store")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, insert_batch_size=32
)
# Text Q&A prompt
text_qa_system_prompt = ChatMessage(
    content=(
        "<<SYS>> You are an expert Q&A system that is trusted around the world.\n"
        "You always answer the query using the provided context information "
        "and never use your prior knowledge. <</SYS>>"
    ),
    role=MessageRole.SYSTEM,
)

text_qa_prompt_tmpl_msgs = [
    text_qa_system_prompt,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------\n"
            "{context_str}\n"
            "---------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Do not include statements like 'According to the provided text' or 'Based "
            "on the provided information' or anything similar in your answer.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

chat_text_qa_prompt = ChatPromptTemplate(message_templates=text_qa_prompt_tmpl_msgs)

query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=env.similarity_top_k,
    text_qa_template=chat_text_qa_prompt,
)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=env.chunk_size, chunk_overlap=env.chunk_overlap)
    ],
    vector_store=vector_store,
)

# Async Qdrant client:
async_client = AsyncQdrantClient(url=env.vector_store_url, port=443)
async_client.set_model(env.embed_model, cache_dir="/storage")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", cache_dir="/storage", use_fp16=True)


def format_prompt(messages: list[Message], family: str = "phi3") -> str:
    prompt = ""
    if family == "llama2":
        prompt = f"<s>[INST] <<SYS>>\n{messages[0].content}\n<</SYS>>\n\n{messages[1].content} [/INST] "
        for message in messages[2:]:
            if message.role == "user":
                prompt += f"\n<s>[INST] {message.content} [/INST] "
            elif message.role == "assistant":
                prompt += f"{message.content} </s>"
            else:
                continue
        return prompt
    elif family == "phi3":
        prompt = f""
        for message in messages[1:]:
            if message.role == "user":
                prompt += f"<|user|>\n{message.content}<|end|>\n<|assistant|>"
            elif message.role == "assistant":
                prompt += f"\n{message.content}<|end|>\n"
            else:
                continue
        return prompt


def process_names(names: list[str]) -> list[str]:
    return names


def process_datetimes(datetimes: list[str]) -> dict[str, pendulum.DateTime]:
    # Predefined strings.
    numbers = (
        "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten|"
        "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        "eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
        "ninety|hundred|thousand)"
    )
    all_numbers = f"(\d+|({numbers}[-\s]?)+)"
    week_day = "(monday|mon|tuesday|tue|wednesday|wed|thursday|thu|friday|fri|saturday|sat|sunday|sun)"
    months = "(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)"
    mhdmy = "(minute|hour|day|week|month|year)"
    time_day = "(morning|afternoon|evening|night)"
    exp1 = "(before|after|earlier|later|ago)"
    exp2 = "(in|on|this|next|following|coming|upcoming|last|previous|since)"
    rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
    iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
    year = "((?<=\s)\d{4}|^\d{4})"

    regxp1 = f"({all_numbers} {mhdmy}s? {exp1})"
    regxp2 = f"({exp2} ({all_numbers} )?({mhdmy}s?|{week_day}|{months}|{time_day}))"

    # Hash function for week days to simplify the grounding task.
    # [Mon..Sun] -> [0..6]
    hashweekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }

    # Hash function for months to simplify the grounding task.
    # [Jan..Dec] -> [1..12]
    hashmonths = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    tenses = {
        "before": -1,
        "earlier": -1,
        "ago": -1,
        "last": -1,
        "previous": -1,
        "this": 0,
        "after": 1,
        "later": 1,
        "next": 1,
        "following": 1,
        "coming": 1,
        "upcoming": 1,
    }

    def convert(time: str, plural: bool = True) -> str:
        if plural:
            if "s" not in time:
                time = f"{time}s"
            return time
        else:
            if "s" in time:
                time = time[:-1]
            return time

    now: pendulum.DateTime = pendulum.now()

    # # Initialization
    # timex_found = []

    # # re.findall() finds all the substring matches, keep only the full
    # # matching string. Captures expressions such as 'number of days' ago, etc.
    # found = re.findall(regxp1, text, re.IGNORECASE)
    # found = [a[0] for a in found if len(a) > 1]
    # for timex in found:
    #     timex_found.append(timex)

    # # Variations of this thursday, next year, etc
    # found = re.findall(regxp2, text, re.IGNORECASE)
    # found = [a[0] for a in found if len(a) > 1]
    # for timex in found:
    #     timex_found.append(timex)

    # # today, tomorrow, etc
    # found = re.findall(rel_day, text, re.IGNORECASE)
    # for timex in found:
    #     timex_found.append(timex)

    # # ISO
    # found = re.findall(iso, text, re.IGNORECASE)
    # for timex in found:
    #     timex_found.append(timex)

    # # Year
    # found = re.findall(year, text, re.IGNORECASE)
    # for timex in found:
    #     timex_found.append(timex)

    datetime_ranges = {}

    # Calculate the new date accordingly
    for timex in datetimes:

        # If numbers are given in words, hash them into corresponding numbers.
        # eg. twenty five days ago --> 25 days ago
        if re.search(numbers, timex, re.IGNORECASE):
            splits = re.split(numbers, timex, 100, re.IGNORECASE)
            if splits[1] == "a":
                splits[1] = "one"
            num = w2n.word_to_num("".join(splits[1:-1]))
            timex = f"{splits[0]}{num}{splits[-1]}"

        # Relative dates
        if re.fullmatch(r"tonight|tonite|today", timex, re.IGNORECASE):
            datetime_ranges[timex] = {
                "start": now.start_of("day"),
                "end": now.end_of("day"),
            }

        elif re.fullmatch(r"yesterday", timex, re.IGNORECASE):
            date = now.add(days=-1)
            datetime_ranges[timex] = {
                "start": date.start_of("day"),
                "end": date.end_of("day"),
            }

        elif re.fullmatch(r"tomorrow", timex, re.IGNORECASE):
            date = now.add(days=+1)
            datetime_ranges[timex] = {
                "start": date.start_of("day"),
                "end": date.end_of("day"),
            }

        # Calculate the offset by taking '\d+' part from the timex.
        elif re.fullmatch(f"\d+ ({mhdmy})s? ({exp1})", timex, re.IGNORECASE):
            (num, time, tense) = re.split(r"\s", timex)
            date = now.add(**{convert(time): tenses[tense.lower()] * int(num)})
            datetime_ranges[timex] = {
                "start": date.start_of(convert(time, False)),
                "end": date.end_of(convert(time, False)),
            }
        elif re.fullmatch(f"({exp2}) \d+ ({mhdmy})s?", timex, re.IGNORECASE):
            (tense, num, time) = re.split(r"\s", timex)
            coef = tenses[tense.lower()]
            date = now.add(**{convert(time): coef * int(num)})
            if coef < 0:
                datetime_ranges[timex] = {"start": date, "end": now}
            else:
                datetime_ranges[timex] = {"start": now, "end": date}

        elif re.fullmatch(f"{exp2} {mhdmy}", timex, re.IGNORECASE):
            (tense, time) = timex.split()
            date = now.add(**{convert(time): tenses[tense.lower()]})
            datetime_ranges[timex] = {
                "start": date.start_of(time),
                "end": date.end_of(time),
            }

        # Weekday in the previous week.
        elif re.fullmatch(f"(last|previous|since) {week_day}", timex, re.IGNORECASE):
            day = hashweekdays[timex.split()[1].lower()]
            date = now.previous(day)
            datetime_ranges[timex] = {
                "start": date.start_of("day"),
                "end": date.end_of("day"),
            }
        # Weekday in the current week.
        elif re.fullmatch(f"(on|this) {week_day}", timex, re.IGNORECASE):
            day = hashweekdays[timex.split()[1].lower()]
            date = now.add(days=(day - now.weekday()))
            datetime_ranges[timex] = {
                "start": date.start_of("day"),
                "end": date.end_of("day"),
            }
        # Weekday in the following week.
        elif re.fullmatch(
            f"(next|following|coming|upcoming) {week_day}", timex, re.IGNORECASE
        ):
            day = hashweekdays[timex.split()[1].lower()]
            date = now.next(day)
            datetime_ranges[timex] = {
                "start": date.start_of("day"),
                "end": date.end_of("day"),
            }

        # Month in the previous year.
        elif re.fullmatch(f"(last|previous|since) {months}", timex, re.IGNORECASE):
            month = hashmonths[timex.split()[1].lower()]
            offset = month - now.month
            date = now.add(years=-1 + (offset < 0), months=(month - now.month))
            datetime_ranges[timex] = {
                "start": date.start_of("month"),
                "end": date.end_of("month"),
            }
        # Month in the current year.
        elif re.fullmatch(f"(in|this) {months}", timex, re.IGNORECASE):
            month = hashmonths[timex.split()[1].lower()]
            date = now.add(months=(month - now.month))
            datetime_ranges[timex] = {
                "start": date.start_of("month"),
                "end": date.end_of("month"),
            }
        # Month in the following year.
        elif re.fullmatch(
            f"(next|following|coming|upcoming) {months}", timex, re.IGNORECASE
        ):
            month = hashmonths[timex.split()[1].lower()]
            offset = month - now.month
            date = now.add(years=1 - (offset > 0), months=offset)
            datetime_ranges[timex] = {
                "start": date.start_of("month"),
                "end": date.end_of("month"),
            }

    if datetime_ranges:
        return max(datetime_ranges.items(), key=lambda x: x[1]["end"] - x[1]["start"])
    else:
        return "", {}


async def process(text: str):
    messages = [
        Message(role="system", content=""),
        Message(
            content=f"""You are skilled in extracting names and datetimes information from a text body.\n"""
            f"""All output must be in valid JSON. Don't add explanation beyond the JSON.\n"""
            f"""Ignore any honorifics like "Prof.", "Ms.", "Dr.", etc.\n"""
            f"""Datetime can be: "last 3 months", "yesterday", "on Sunday", "since 2012", "next July", "May 06", "August 17, 2024", etc\n\n"""
            f"""Example:\n"""
            f"""Text: "What did Sami Hakkinen mention about two weeks earlier?"\n{{"names": ["Sami Hakkinen"], "datetimes": ["2 weeks earlier"]}}\n\n"""
            f"""Text: "In april, was there any report sent to ms.ling?"\n{{"names": ["Ling"], "datetimes": ["in April"]}}\n\n"""
            f"""Text: "Summarize this year budget plan that was discussed a day ago, or on March 24, 2022"\n{{"names": [], "datetimes": ["1 day ago", "March 24, 2022"]}}\n\n"""
            f"""Text: "Explain again how Jimin Young and Mr.Safeer configures the infrastructure."\n{{"names": ["Jimin Young", "Safeer"], "datetimes": []}}\n\n"""
            f"""Text: "What was discussed about the next exploding sun event?"\n{{"names": [], "datetimes": []}}\n\n"""
            f'''Extract all names and datetimes in the following text:\nText: "{text}"'''
        ),
    ]
    result = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url="https://development-llm.dbinno.com/completion",
            json={
                "prompt": format_prompt(messages, family="phi3"),
                "stream": True,
                "n_predict": 512,
                "temperature": 0.1,
            },
        ) as response:
            async for chunks in response.content.iter_chunked(2048):
                chunks = chunks.decode("utf-8").strip().split("\n\n")
                for chunk in chunks:
                    try:
                        data = json.loads(chunk[6:])
                        if "\n\n" in result or data["content"] in [
                            "<|end|>",
                            "<|endoftext|>",
                        ]:
                            break
                        else:
                            result += data["content"]
                    except Exception as e:
                        break

    result = json.loads(result)
    datetime, drange = process_datetimes(result["datetimes"])
    return {
        "names": process_names(result["names"]),
        "datetime": datetime,
        "range": drange,
    }


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI Assistant</title>
        <link
        rel="stylesheet"
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
        />
        <style>
        body {
            margin: 0;
            padding: 0;
        }
        .container {
            display: flex;
            flex-direction: row;
            height: 100vh;
        }
        .column {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0px 10px 0px 10px;
        }
        .messages {
            overflow-y: auto;
            flex-grow: 1;
        }
        .message-container {
            margin-bottom: 10px;
            white-space: pre-wrap;
            color: #fff;
            border-radius: 10px;
            padding: 10px;
        }
        .user-message {
            background-color: #007bff;
        }
        .ai-message {
            background-color: #28a745;
        }
        .document-message {
            background-color: #949400;
        }
        .input-container {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .file-upload-column {
            border-right: 1px solid #ccc;
        }
        </style>
    </head>
    <body>
        <div class="container">
        <div class="column file-upload-column">
            <div>
            <h3>Document Upload</h3>
            <p>
                The uploaded documents will be stored in database and "learned" by
                AI.
            </p>
            <input type="file" id="fileInput" multiple />
            <button id="submitBtn" class="btn btn-primary mt-2">Submit</button>
            <ul id="fileList"></ul>
            </div>
            <hr />
            <h3>Retrieved Information</h3>
            <p>
            This is just for understanding what information the AI is able to
            retrieve from the database. In practice, these are not displayed for
            end user.
            </p>
            <div id="retrievedDocuments" class="messages"></div>
        </div>
        <div class="column file-upload-column">
            <h3>Ask Question About Uploaded Documents</h3>
            <div class="messages" id="ragMessages">
            <!-- Messages will be added dynamically here -->
            </div>
            <div class="input-container">
            <input
                type="text"
                class="form-control"
                id="ragInput"
                placeholder="Ask AI anything about your documents ..."
            />
            </div>
        </div>
        <div class="column">
            <h3>General Purpose AI Assistant</h3>
            <p>
            Suitable for general tasks that don't need information from uploaded
            documents.
            </p>
            <div class="messages" id="aiMessages">
            <!-- Messages will be added dynamically here -->
            </div>
            <div class="input-container">
            <input
                type="text"
                class="form-control"
                id="aiInput"
                placeholder="Ask AI anything ..."
            />
            </div>
        </div>
        </div>

        <script>
        const ragMessagesContainer = document.getElementById("ragMessages");
        const aiMessagesContainer = document.getElementById("aiMessages");
        const ragInput = document.getElementById("ragInput");
        const aiInput = document.getElementById("aiInput");
        const aiMessages = [
            {
            role: "system",
            content: "<<SYS>> You are a helpful assistant. <</SYS>>",
            },
        ];

        async function retrieve(message) {
            const url = "/retrieve";

            const data = {
            content: message,
            };

            try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                "Content-Type": "application/json",
                },
                body: JSON.stringify(data),
            });
            const documents = await response.json();
            const retrievedDocuments =
                document.getElementById("retrievedDocuments");
            retrievedDocuments.innerHTML = "";

            documents.forEach((doc) => {
                const element = document.createElement("p");
                element.classList.add("message-container", "document-message");
                element.textContent = doc.content;
                retrievedDocuments.appendChild(element);
            });
            } catch (error) {
            console.error(error);
            }
        }

        async function ragChat(message) {
            const url = "/chat";

            const data = {
            content: message,
            };

            const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            const messageElement = document.createElement("div");
            messageElement.classList.add("message-container", "ai-message");
            ragMessagesContainer.appendChild(messageElement);

            let chunks = "";

            while (true) {
            const { done, value } = await reader.read();

            if (done) break;
            try {
                chunks += decoder.decode(value, {
                stream: true,
                });
                messageElement.textContent = chunks;
                scrollToBottom(ragMessagesContainer);
            } catch (error) {
                continue;
            }
            }
        }

        async function aiChat(input) {
            const url = "https://development-llm.dbinno.com/v1/chat/completions";

            aiMessages.push({
            role: "user",
            content: input,
            });

            const data = {
            messages: aiMessages,
            stream: true,
            max_tokens: 1024,
            temperature: 0.8,
            };

            const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const messageElement = document.createElement("div");
            messageElement.classList.add("message-container", "ai-message");
            aiMessagesContainer.appendChild(messageElement);
            let answer = "";

            while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            try {
                const chunks = decoder
                .decode(value, {
                    stream: true,
                })
                .trim()
                .replace(/[\\r\\n]+/g, "\\r\\n")
                .split("\\r\\n");

                for (const chunk of chunks) {
                const content = JSON.parse(chunk.substring(6)).choices[0].delta;
                if (content.hasOwnProperty("content")) {
                    answer += content.content;
                    messageElement.textContent = answer;
                    scrollToBottom(aiMessagesContainer);
                }
                }
            } catch (error) {
                console.error("===>", error);
                continue;
            }
            }

            aiMessages.push({
            role: "assistant",
            content: answer.slice(0, -4),
            });
        }

        function scrollToBottom(container) {
            container.scrollTop = container.scrollHeight;
        }

        ragInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            const message = ragInput.value.trim();
            if (message !== "") {
                const messageElement = document.createElement("div");
                messageElement.classList.add("message-container", "user-message");
                messageElement.textContent = message;
                ragMessagesContainer.appendChild(messageElement);
                scrollToBottom(ragMessagesContainer);
                retrieve(message);
                ragChat(message);
                ragInput.value = "";
            }
            }
        });

        aiInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            const message = aiInput.value.trim();
            if (message !== "") {
                const messageElement = document.createElement("div");
                messageElement.classList.add("message-container", "user-message");
                messageElement.textContent = message;
                aiMessagesContainer.appendChild(messageElement);
                scrollToBottom(aiMessagesContainer);
                aiChat(message);
                aiInput.value = "";
            }
            }
        });

        // File upload handling
        const fileInput = document.getElementById("fileInput");
        const fileList = document.getElementById("fileList");
        const submitBtn = document.getElementById("submitBtn");

        submitBtn.addEventListener("click", async function () {
            submitBtn.disabled = true;
            submitBtn.textContent = "Uploading...";

            const files = Array.from(fileInput.files);

            const formData = new FormData();
            files.forEach((file) => {
            formData.append("files", file);
            const listItem = document.createElement("li");
            listItem.textContent = file.name;
            fileList.appendChild(listItem);
            });

            try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            } catch (error) {
            console.error("Error uploading files:", error);
            } finally {
            // Enable the button and change its text back to "Submit"
            submitBtn.disabled = false;
            submitBtn.textContent = "Submit";
            }
        });
        </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_files(files: list[UploadFile]):
    filepaths = []
    for file in files:
        filepath = f"/storage/{file.filename}"
        filepaths.append(filepath)
        with open(filepath, "wb") as f:
            f.write(await file.read())
    documents = SimpleDirectoryReader(input_files=filepaths).load_data()
    nodes = pipeline.run(documents=documents)
    index.insert_nodes(nodes)
    return {"filenames": [file.filename for file in files]}


@app.post("/retrieve")
async def retrieve(message: Message) -> list[Document]:
    nodes = query_engine.retrieve(QueryBundle(query_str=message.content))
    data = [Document(content=node.text) for node in nodes]
    return data


@app.post("/chat")
async def chat(message: Message):

    def stream():
        response = query_engine.query(message.content)
        for text in response.response_gen:
            yield text

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/summarize")
async def summarize(message: Message):
    prompt = "Summarize the given text into a succinct, accurate single paragraph, excluding casual language and retaining only pertinent information:"
    messages = [
        Message(
            role="system",
            content="You specialize in summarizing text. Your summarization should be in a single paragraph. The summarized paragraph should be short, succinct, accurate and contains only relevant information. Also remove casual conversation words.",
        ),
        Message(
            role="user",
            content=f"""{prompt}\nHey guys! For KPMG tenant only, I can see E11000 duplicate key error collection: kpmg2.features index: displayname_1 dup key: {{ displayname: "Chat Message retention" }} Have to dig out why the document is being recreated in this instance""",
        ),
        Message(
            role="assistant",
            content="""An E11000 duplicate key error occurred for a KPMG tenant with the collection 'kpmg2.features'. The index 'displayname_1' encountered a duplicate key for the display name "Chat Message retention". Investigation is needed to determine why the document is being recreated in this instance.""",
        ),
        Message(
            role="user",
            content=f"""{prompt}\nThis is ok. Of course - my consideration was from user's perspective. So the invited #One user will receive an email and then they attempt to sign in to #One and are taken to DB IdP, BUT as they have not set their name (depends if we set it in CAC - this seems to be optional in other systems - if admin doesn't set it, only email, then user is asked to enter these) or pwd i.e. they need to register. After registering either go directly to #One or ask to sign in with given pwd. We should consider if the pwd can be set to remember in browser in the UI where the password is created.""",
        ),
        Message(
            role="assistant",
            content="""The invited #One user receives an email and tries to sign in to #One via DB IdP. If their name is unset (depends on CAC settings), they're prompted to enter it along with a password for registration. After registering, they can access #One or sign in using the provided password, with a possible browser password remember feature.""",
        ),
        Message(
            role="user",
            content=f"""{prompt}\nGood morning Hallie. Regarding the SCIM or OIDC or SAML integrations towards IdPs. Most IdPs support some additional data that can be received alongside the identity information. It would be good to consider getting the Security Group information at the minimum. It would be absolutely great, if we could also get additional information like phone number, maybe position/title etc., but this should be considered optional and such that can also be managed from within the CIAM and maybe even products.""",
        ),
        Message(
            role="assistant",
            content="""For SCIM, OIDC, or SAML integrations with IdPs, consider retrieving Security Group information as a minimum from IdPs supported data. Optionally, extra data like phone numbers and position titles can also be managed within the CIAM system or products.""",
        ),
    ]
    summarize = ""
    messages.append(
        Message(
            role="user",
            content=f"""{prompt}\n{message.content}""",
        )
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url="https://development-llm.dbinno.com/completion",
            json={
                "prompt": format_prompt(messages, family="phi3"),
                "stream": True,
                "n_predict": 512,
                "temperature": 0.1,
            },
        ) as response:
            async for chunks in response.content.iter_chunked(2048):
                chunks = chunks.decode("utf-8").strip().split("\n\n")
                for chunk in chunks:
                    try:
                        data = json.loads(chunk[6:])
                        if data["content"] not in [
                            "<|end|>",
                            "<|endoftext|>",
                        ]:
                            summarize += data["content"]
                    except Exception as e:
                        break
    return summarize.strip()


@app.post("/embedding")
async def embedding(embedding: Embedding):
    original = re.sub(r"\s+", " ", embedding.content)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url="http://rag.default.svc.cluster.local/summarize",
            json={"content": original},
        ) as response:
            summarize = await response.text()
    content = f"{embedding.author} said:\n{summarize[1:-1]}"
    now = pendulum.now().to_rfc3339_string()
    response = await async_client.add(
        collection_name="One",
        metadata=[
            {
                "author": embedding.author.split() + [embedding.author],
                "boundary": embedding.boundary,
                "created_at": now,
                "updated_at": now,
                "original": original,
            }
        ],
        documents=[content],
        ids=[str(uuid.uuid4())],
    )
    return response


@app.post("/assistant")
async def assistant(messages: Messages):

    query = messages.messages[-1].content

    # Compose data filters to filter as many irrelevant data as possible:
    filters = [
        models.FieldCondition(
            key="boundary",
            match=models.MatchAny(any=[messages.user]),
        ),
    ]

    extraction = await process(query)

    if extraction["names"]:
        filters.append(
            models.FieldCondition(
                key="author",
                match=models.MatchAny(any=extraction["names"]),
            )
        )

    if extraction["range"]:
        filters.append(
            models.FieldCondition(
                key="updated_at",
                range=models.DatetimeRange(
                    gt=None,
                    gte=extraction["range"]["start"].to_rfc3339_string(),
                    lt=extraction["range"]["end"].to_rfc3339_string(),
                    lte=None,
                ),
            )
        )

    # Query vector database for relevant documents:
    documents = await async_client.query(
        collection_name="One",
        query_text=query,
        limit=10,
        score_threshold=0.3,
        query_filter=models.Filter(must=filters),
    )
    for d in documents:
        print("-", d.score, d.document[:100])

    # Rerank the retrieved documents:
    if len(documents) > 0:
        scores = reranker.compute_score(
            [[query, document.document] for document in documents], normalize=True
        )
        if type(scores) != list:
            scores = [scores]
        scores = sorted(zip(scores, documents), reverse=True)
        print("=========================")
        for d in scores:
            print("-", d[0], d[1].score, d[1].document[:100])

        threshold = 0.5 if scores[0][0] >= 0.5 else 1e-3
        scores = list(filter(lambda x: x[0] >= threshold, scores))[:5]
        print("=========================")
        print(scores)
        if len(scores) > 0:
            messages.messages[-1].content = (
                f"""Here's the list of topics discussed by {', '.join(extraction["names"])} {extraction["datetime"]}:\n"""
                f"""{''.join([f'- {s[1].document.split(chr(10))[-1]}{chr(10)}' for s in scores])}\n"""
                f"""Do not include statements like 'According to the provided text' or 'Based """
                f"""on the provided information' or anything similar in your answer.\n"""
                f"""Answer this question: {query}"""
            )
        else:
            messages.messages[-1].content = (
                f"""No relevant discussion by {', '.join(extraction["names"])} {extraction["datetime"]} was found.\n"""
                f"""Answer this question: {query}"""
            )
    else:
        messages.messages[-1].content = (
            f"""No relevant discussion by {', '.join(extraction["names"])} {extraction["datetime"]} was found.\n"""
            f"""Answer this question: {query}"""
        )
    print(format_prompt(messages.messages, family="phi3"))

    # Final call to LLM:
    async def stream():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://development-llm.dbinno.com/completion",
                json={
                    "prompt": format_prompt(messages.messages, family="phi3"),
                    "stream": True,
                    "n_predict": messages.max_tokens,
                    "temperature": messages.temperature,
                },
            ) as response:
                async for chunks in response.content.iter_chunked(2048):
                    chunks = chunks.decode("utf-8").strip().split("\n\n")
                    for chunk in chunks:
                        try:
                            data = json.loads(chunk[6:])
                            if data["content"] not in ["<|end|>", "<|endoftext|>"]:
                                yield data["content"]
                        except Exception as e:
                            break

    return StreamingResponse(stream(), media_type="text/event-stream")
