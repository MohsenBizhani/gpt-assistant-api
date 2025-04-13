from fastapi import FastAPI, Form, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from database import initialize_db, get_user, add_user, get_threads, add_thread, remove_thread, add_message, get_messages
from openai import OpenAI
from contextlib import asynccontextmanager
import time
import re
from markdown import markdown
from conf import openai_api, assistant_id
from fastapi.responses import StreamingResponse, JSONResponse

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ 
    Lifespan manager for database initialization.
    Starts when app starts, initializes database connection.
    """
    await initialize_db()
    yield

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_errors(request: Request, call_next):
    """
    Global error handling middleware.
    Logs all exceptions and returns 500 status for unhandled errors.
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"error": "Internal Server Error", "status_code": 500}

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key here
ASSISTANT_ID = assistant_id

def startBot():
    """
    Initializes OpenAI client and retrieves assistant.
    Returns: Assistant ID string
    """
    client = OpenAI(api_key=openai_api)
    assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
    return assistant.id

def startThread(prompt):
    """
    Creates a new conversation thread with initial message.
    Parameters: prompt - initial user message
    Returns: OpenAI thread ID string
    """
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(api_key=openai_api)
    thread = client.beta.threads.create(messages=messages)
    return thread.id

def runAssistant(thread_id, assistant_id):
    """
    Starts assistant processing on a thread.
    Parameters: thread_id - existing thread ID, assistant_id - OpenAI assistant ID
    Returns: OpenAI run ID string
    """
    client = OpenAI(api_key=openai_api)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    return run.id

def checkRunStatus(thread_id, run_id):
    """
    Checks status of an assistant run.
    Returns: Status string ('completed', 'failed', etc.)
    """
    client = OpenAI(api_key=openai_api)
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status

def retrieveThread(thread_id):
    """
    Retrieves and formats messages from a thread.
    Converts markdown to HTML and handles image attachments.
    Returns: Formatted HTML content string or None
    """
    client = OpenAI(api_key=openai_api)
    thread_messages = client.beta.threads.messages.list(thread_id=thread_id)
    list_messages = thread_messages.data

    logging.error(f"Number of messages: {len(list_messages)}, Messages: {list_messages}")

    for message in list_messages:
        if message.content:
            for content_block in message.content:
                if content_block.type == 'text':
                    message_text = content_block.text.value
                    logging.error(f"Message Text: {message_text}")
                    html_content = markdown(message_text)
                    return html_content
                elif content_block.type == 'image_file':
                    image_file = content_block.image_file
                    logging.error(f"Image File ID: {image_file.file_id}")
                    # Handle image processing if needed
    logging.error("No valid content found in any message.")
    return None

def addMessageToThread(thread_id, prompt1):
    """
    Adds a user message to an existing thread.
    Parameters: thread_id - target thread ID, prompt1 - message content
    """
    client = OpenAI(api_key=openai_api)
    thread_message = client.beta.threads.messages.create(thread_id=thread_id, role='user', content=prompt1)
    logging.error(f'trying to add {prompt1} to messages in thread {thread_id}')
    add_message(thread_id, 'user', prompt1)

UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

from io import BytesIO
from fastapi.responses import StreamingResponse




@app.get("/threads/{user_id}/{thread_id}/messages-with-files")
async def get_messages_with_files(user_id: str, thread_id: str):
    """
    Retrieves messages with file attachments in a thread.
    Returns: List of messages with file download links
    """

    try:
        threads = get_threads(user_id)

        if thread_id not in [t["thread_id"] for t in threads]:
            return JSONResponse(status_code=404, content={"detail": "Thread Not Found!"})

        messages = get_messages(thread_id)
        if not messages:
            return {"message": "No messages found in this thread."}

        messages_with_files = [
            {"message_id": message_time, "file_link": f"https://api.orbexai.com/get-file/{image_id}"}
            for _, _, image_id, message_time in messages if image_id
        ]

        if not messages_with_files:
            return {"message": "No messages with files found in this thread."}

        return {
            "thread_id": thread_id,
            "messages_with_files": messages_with_files,
            "total_files": len(messages_with_files)
        }

    except Exception as e:
        logging.error(f"Error retrieving messages with files: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving messages with files.")

from base64 import b64encode
from fastapi.responses import JSONResponse

@app.get("/threads/{user_id}/{thread_id}/download-files")
async def download_files(user_id: str, thread_id: str):
    """
    Downloads all files from a thread as base64 encoded content.
    Returns: JSON with file data array
    """
    try:
        threads = get_threads(user_id)

        if thread_id not in [t["thread_id"] for t in threads]:
            return JSONResponse(status_code=404, content={"detail": "Thread Not Found!"})

        messages = get_messages(thread_id)
        if not messages:
            return {"message": "No messages found in this thread."}

        file_responses = []
        client = OpenAI(api_key=openai_api)

        for _, _, image_id, message_time in messages:
            if image_id:
                try:
                    # Fetch file metadata
                    file_metadata = client.files.retrieve(image_id)
                    file_extension = file_metadata.filename.split('.')[-1] if hasattr(file_metadata, "filename") else "bin"
                    file_name = f"{message_time}.{file_extension}"

                    # Fetch file content
                    response = client.files.content(image_id)
                    file_content = await response.aread()

                    # Encode file content in base64
                    encoded_content = b64encode(file_content).decode("utf-8")

                    # Append the file details to the response list
                    file_responses.append({
                        "message_id": message_time,
                        "file_name": file_name,
                        "file_content": encoded_content,
                        "mime_type": file_metadata.mimetype if hasattr(file_metadata, "mimetype") else "application/octet-stream",
                    })
                except Exception as e:
                    logging.error(f"Error fetching file {image_id}: {e}")

        if not file_responses:
            return {"message": "No files available for download."}

        return JSONResponse(content={"files": file_responses})

    except Exception as e:
        logging.error(f"Error downloading files: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while downloading files.")





@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file uploads to OpenAI servers.
    Accepts: JPEG, PNG, GIF, WEBP
    Returns: OpenAI file ID
    """

    try:
        client = OpenAI(api_key=openai_api)

        valid_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid image type. Supported types are: .jpeg, .jpg, .png, .gif, .webp")

        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        try:
            with open(file_path, 'rb') as f:
                file_response = client.files.create(
                    file=f,
                    purpose='vision'
                )
                file_id = file_response.id
        finally:
            os.remove(file_path)

        return {"file_id": file_id}
    except Exception as e:
        logging.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while uploading the file.")



@app.get("/get-file/{file_id}")
async def get_file(file_id: str):
    """
    Retrieves a file from OpenAI by ID.
    Returns: File content as stream with proper headers
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api)

        # Fetch the file metadata (to get the filename and extension)
        file_metadata = client.files.retrieve(file_id)
        file_name = file_metadata.filename if hasattr(file_metadata, "filename") else "downloaded_file"

        # Fetch the file content
        response = client.files.content(file_id)

        # Extract raw bytes from the response
        file_content = await response.aread()  # Use 'aread()' to read content asynchronously

        # Create a BytesIO stream from the content
        file_stream = BytesIO(file_content)

        # Return the content as a streaming response with a proper filename
        headers = {"Content-Disposition": f"attachment; filename={file_name}"}
        return StreamingResponse(file_stream, media_type="application/octet-stream", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

from database import get_latest_user_message, update_message_content, get_latest_image_id_in_thread

@app.post("/regenerate")
async def regenerate(user_id: str = Form(...), thread_id: str = Form(...), new_prompt: str = Form(None)):
    """
    Regenerates assistant response for last message.
    Allows editing previous prompt or using original.
    Returns: New formatted response with metadata
    """
    try:
        logging.debug(f"Starting regeneration process for user_id: {user_id}, thread_id: {thread_id}, new_prompt: {new_prompt}")

        # 1.1 Check if the user exists
        user = get_user(user_id)
        if not user:
            logging.warning(f"User not found: {user_id}")
            return {"status": 404, "detail": "User not found."}
        
        # 1.2 Check if the thread exists for the user
        threads = get_threads(user_id)
        if not any(t["thread_id"] == thread_id for t in threads):
            logging.warning(f"Thread not found for user: {user_id}, thread_id: {thread_id}")
            return {"status": 404, "detail": "Thread not found for this user."}
        
        # Fetch the latest user message in the thread
        latest_user_message = get_latest_user_message(thread_id)
        if not latest_user_message:
            logging.warning(f"No user messages found in thread: {thread_id}")
            return {"status": 404, "detail": "No user messages found in this thread."}
        
        user_message_id, original_prompt = latest_user_message
        logging.debug(f"Latest user message found: message_id={user_message_id}, original_prompt={original_prompt}")
        
        # Fetch the image_id from the latest user message
        hasImage = False
        image_id = get_latest_image_id_in_thread(thread_id=thread_id)
        if image_id:
            hasImage = True
        logging.debug(f"Latest image_id in thread: {image_id}")
        
        # If the user is asking about an image, fetch the latest image_id in the thread
        if not image_id and "image" in (new_prompt or original_prompt).lower():
            image_id = get_latest_image_id_in_thread(thread_id)
            logging.debug(f"Image ID fetched based on prompt: {image_id}")
        
        # If a new prompt is provided, update the latest user message
        if new_prompt:
            logging.debug(f"Updating message content with new prompt: {new_prompt}")
            update_message_content(user_message_id, new_prompt)
            prompt = new_prompt
        else:
            prompt = original_prompt
            logging.debug(f"Using original prompt: {prompt}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api)
        
        # Add the updated prompt to the thread, including the image if it exists

        

        
        if image_id:
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_file", "image_file": {"file_id": image_id}}
            ]
            logging.debug(f"Message content includes image with file_id: {image_id}")
        else:
            message_content = [{"type": "text", "text": prompt}]


        
        logging.debug(f"Creating message in thread: thread_id={thread_id}, role=user, content={message_content}")
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )
        
        # Run the assistant to generate a response
        logging.debug(f"Running assistant for thread_id: {thread_id}, assistant_id: {assistant_id}")
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logging.debug(f"Assistant run started: run_id={run.id}")
        
        # Wait for the assistant to complete the run
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logging.debug(f"Checking run status: status={run_status.status}")
            if run_status.status == "completed":
                logging.debug("Assistant run completed successfully.")
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                logging.error(f"Assistant run failed or was cancelled: status={run_status.status}")
                return {"status": 500, "detail": "Assistant run failed or was cancelled."}
            time.sleep(2)  # Polling interval
        
        # Retrieve the assistant's response
        logging.debug(f"Retrieving messages from thread: thread_id={thread_id}")
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        assistant_response = None
        
        for message in messages.data:
            if message.role == "assistant" and message.content:
                assistant_response = message.content[0].text.value
                logging.debug(f"Assistant response found: {assistant_response}")
                break
        
        if not assistant_response:
            logging.error("No response generated by the assistant.")
            return {"status": 500, "detail": "No response generated by the assistant."}
        
        # Add the assistant's response to the database, linking it to the user message via `reply_to`
        logging.debug(f"Adding assistant response to database: thread_id={thread_id}, reply_to={user_message_id}")
        add_message(thread_id, 'assistant', assistant_response, reply_to=user_message_id)
        
        # Fetch the thread title for the response
        thread_title = next((t["title"] for t in threads if t["thread_id"] == thread_id), "Untitled Thread")
        logging.debug(f"Thread title: {thread_title}")

        html_format = markdown(assistant_response)
        
        # Return the response in the same format as the /ask route, including `reply_to`
        return {
            "thread_id": thread_id,
            "thread_title": thread_title,
            "UM_id": user_message_id,
            "hasImage": hasImage,
            "message": html_format,
            "file_id": image_id,  # Include the image_id in the response
            "reply_to": user_message_id  # Include the `reply_to` field in the response
        }
    except Exception as e:
        logging.error(f"Error during regeneration: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while regenerating the response.")




@app.post("/ask")
async def ask(
    user_id: str = Form(...),
    prompt: str = Form(None),
    thread_id: str = Form(None),
    file_id: str = Form(None),
    reply: int = Form(None)
):
    """
    Main Q&A endpoint. Handles:
    - New conversations (creates thread)
    - Existing threads (continues conversation)
    - File attachments
    - Message replies
    Returns: Formatted response with thread metadata
    """
    try:
        client = OpenAI(api_key=openai_api)
        user = get_user(user_id)
        assistant_id = startBot()

        # Check if both prompt and file_id are empty
        if prompt is None and file_id is None:
            return {"status": 501, "detail": "prompt and file_id cannot be both empty"}
        
        raw_prompt = prompt

        # Handle replies (if reply is provided)
        if reply is not None:
            messages = get_messages(thread_id)
            message_content = next((msg[2] for msg in messages if msg[0] == reply), None)
            if message_content:
                prompt = f"پیام قبلی: {message_content}\n\n{prompt}"
            else:
                logging.warning(f"Message with id {reply} not found.")

        # Default prompt if none is provided
        if prompt is None:
            prompt = "اين تصوير رو آناليز کن"

        # Add user if they don't exist
        if not user:
            add_user(user_id)

        # Create a new thread if thread_id is not provided
        if not thread_id:
            thread_id = startThread(prompt)
            title = generate_title(prompt)
            add_thread(user_id, thread_id, title)
            if not file_id:
                user_message_id = add_message(thread_id, 'user', raw_prompt, reply_to=reply)
            else:
                user_message_id = add_message(thread_id, 'user', raw_prompt, image_id=file_id, reply_to=reply)
        else:
            # Fetch existing thread details
            threads = get_threads(user_id)
            existing_thread = next((t for t in threads if t["thread_id"] == thread_id), None)
            if existing_thread:
                title = existing_thread["title"]
            else:
                raise HTTPException(status_code=404, detail="Thread not found for this user.")

            # Fetch the latest image_id if the user is asking about an image
            if not file_id and "image" in prompt.lower():
                file_id = get_latest_image_id_in_thread(thread_id)

            # Add the user message to the thread
            if not file_id:
                user_message_id = add_message(thread_id, 'user', raw_prompt, reply_to=reply)
            else:
                user_message_id = add_message(thread_id, 'user', raw_prompt, image_id=file_id, reply_to=reply)

        has_image = False

        # Create the message content with or without the file_id
        if file_id:
            has_image = True
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_file", "image_file": {"file_id": file_id}}
            ]
            logging.debug(f"Creating message with file_id: {file_id}")
            response = client.beta.threads.messages.create(
                thread_id=thread_id,
                role='user',
                content=message_content
            )
            logging.debug(f"OpenAI API Response: {response}")
        else:
            # Add the user message without file_id
            response = client.beta.threads.messages.create(
                thread_id=thread_id,
                role='user',
                content=[{"type": "text", "text": prompt}]
            )

        # Run the assistant and wait for completion
        run_id = runAssistant(thread_id, assistant_id)
        while checkRunStatus(thread_id, run_id) != 'completed':
            if checkRunStatus(thread_id, run_id) in ['failed', 'expired', 'requires_action', 'cancelled']:
                return {"message": "لطفا دوباره امتحان کنید"}
            time.sleep(2.5)

        # Retrieve the assistant's response
        html_content = retrieveThread(thread_id)
        if html_content:
            add_message(thread_id, 'assistant', html_content, reply_to=user_message_id)

        # Return the response
        return {
            "thread_id": thread_id,
            "thread_title": title,
            "UM_id": user_message_id,
            "message": html_content,
            "hasImage": has_image,
            "file_id": file_id,
            "reply_to": user_message_id
        }
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
    


@app.get("/threads/{user_id}")
async def list_threads(user_id: str, page: int = 1, page_size: int = 10):
    """
    Lists user's conversation threads with pagination.
    Returns: Paginated thread list with metadata
    """
    try:
        threads = get_threads(user_id)
        if not threads:
            return {"message": "No threads found for this user."}

        # Sort threads based on updated_at
        sorted_threads = sorted(threads, key=lambda x: x["updated_at"], reverse=True)

        start = (page - 1) * page_size
        end = start + page_size
        paginated_threads = sorted_threads[start:end]

        return {
            "threads": paginated_threads,
            "page": page,
            "page_size": page_size,
            "total_threads": len(threads)
        }
    except Exception as e:
        logging.error(f"Error retrieving threads: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving threads.")


import requests
def delete_openai_thread(thread_id):
    """
    Deletes a thread using the OpenAI API.

    Parameters:
    - api_key (str): Your OpenAI API key.
    - thread_id (str): The ID of the thread to delete.

    Returns:
    - dict: A dictionary containing the response from the API.
    """
    url = f'https://api.openai.com/v1/threads/{thread_id}'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api}',
        'OpenAI-Beta': 'assistants=v2'
    }

    response = requests.delete(url, headers=headers)

    if response.status_code == 200:
        return True
    else:
        logging.error(f"Failed to delete thread. Status code: {response.status_code}")
        return response


@app.delete("/threads/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    """
    Deletes a conversation thread.
    Removes from both OpenAI and local database.
    """
    try:
        # Fetch the list of threads for the user
        threads = get_threads(user_id)
        
        # Check if the thread exists for the user
        if not any(t["thread_id"] == thread_id for t in threads):
            return {"status_code": 404, "detail": "Thread Not Found"}

        # Remove the thread
        res = delete_openai_thread(thread_id=thread_id)
        if res:
            if(remove_thread(thread_id)):
                return {"message": "Thread deleted successfully."}
    except Exception as e:
        logging.error(f"Error deleting thread: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting the thread.")


@app.get("/threads/{user_id}/{thread_id}")
async def get_thread_messages(user_id: str, thread_id: str):
    """
    Retrieves all messages in a thread.
    Returns: Complete message history with metadata
    """
    try:
        # Fetch the user's threads
        threads = get_threads(user_id)
        
        # Check if the thread exists for the user
        if not any(t["thread_id"] == thread_id for t in threads):
            return {"status_code": 404, "detail": "Thread Not Found!"}
        
        # Fetch messages for the thread
        messages = get_messages(thread_id)
        if not messages:
            return {"message": "No messages found in this thread."}
        
        # Format and return all messages with reply_to
        formatted_messages = [
            {
                "message_id": message_id,
                "role": role,
                "content": content,
                "hasImage": True if image_id else False,
                "image_link": f"https://api.orbexai.com/get-file/{image_id}" if image_id else None,
                "reply_to": reply_to
            }
            for message_id, role, content, image_id, reply_to in messages
        ]
        
        return {
            "messages": formatted_messages,
            "total_messages": len(messages)
        }
    except Exception as e:
        logging.error(f"Error retrieving messages from thread: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving messages.")



def generate_title(prompt, char_limit=30):
    """
    Generates thread title from prompt.
    Truncates long prompts and adds ellipsis.
    """
    # Check if the prompt contains spaces
    if ' ' in prompt:
        # Split the prompt into words and take the first five
        return ' '.join(prompt.split()[:5]) + '...'
    else:
        # If no spaces, limit the title to the specified character limit
        return prompt[:char_limit] + '...'

def remove_pattern(text):
    """
    Cleans special characters from OpenAI responses.
    Converts markdown to HTML.
    """
    modified_text = re.sub(r'【*', '', text)
    return markdown(modified_text)
