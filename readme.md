# GPT Assistant API

A FastAPI-based REST API for managing conversations with OpenAI's GPT Assistant. This project provides a robust backend for handling chat threads, file uploads, and message management with OpenAI's Assistant API.

## Features

- 🤖 OpenAI GPT Assistant Integration
- 📁 File Upload Support (JPEG, PNG, GIF, WEBP)
- 💬 Thread Management
- 🔄 Message Regeneration
- 📝 Message History
- 🖼️ Image Analysis Support
- 🔗 Reply Threading
- 📊 Pagination Support

## Tech Stack

- FastAPI
- OpenAI API
- SQLite
- Python 3.x
- Markdown
- Regex

## Prerequisites

- Python 3.7+
- OpenAI API Key
- OpenAI Assistant ID

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-assistant-api.git
cd gpt-assistant-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `conf.py` file with your OpenAI credentials:
```python
openai_api = 'your-openai-api-key'
assistant_id = 'your-assistant-id'
```

4. Initialize the database:
```bash
python database.py
```

5. Start the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### User & Thread Management

- `GET /threads/{user_id}` - List user's conversation threads
- `DELETE /threads/{user_id}/{thread_id}` - Delete a thread
- `GET /threads/{user_id}/{thread_id}` - Get thread messages

### Message Operations

- `POST /ask` - Send a message or start a new conversation
- `POST /regenerate` - Regenerate assistant response
- `GET /threads/{user_id}/{thread_id}/messages-with-files` - Get messages with file attachments

### File Operations

- `POST /upload-file` - Upload a file
- `GET /get-file/{file_id}` - Download a file
- `GET /threads/{user_id}/{thread_id}/download-files` - Download all files from a thread

## Project Structure

```
gpt-assistant-api/
├── main.py           # FastAPI application and routes
├── database.py       # Database operations and schema
├── conf.py          # Configuration and API keys
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Database Schema

### Users Table
- `user_id` (VARCHAR) - Primary Key

### Threads Table
- `thread_id` (VARCHAR) - Primary Key
- `user_id` (VARCHAR) - Foreign Key
- `title` (TEXT)
- `created_at` (DATETIME)
- `updated_at` (DATETIME)

### Messages Table
- `message_id` (INTEGER) - Primary Key
- `thread_id` (VARCHAR) - Foreign Key
- `role` (TEXT)
- `content` (TEXT)
- `image_id` (VARCHAR)
- `message_time` (DATETIME)
- `reply_to` (INTEGER) - Foreign Key

## Security Considerations

1. API keys are stored in a separate configuration file
2. CORS middleware with configurable origins
3. Error handling and logging
4. Input validation for file uploads
5. Secure file handling

## Error Handling

The API includes comprehensive error handling:
- Global error middleware
- Specific error responses for various scenarios
- Detailed logging

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT Assistant API
- FastAPI framework
- All contributors and users of this project
