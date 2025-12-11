def extract_user_query(full_request: dict) -> str:
    """
    Extracts the meaningful user query from the full request object.
    Filters out 'For context:' messages which are usually injected history/tool outputs.
    Returns the last few meaningful user messages joined together.
    """
    try:
        if not full_request or 'contents' not in full_request:
            return "N/A"
            
        user_messages = []
        for content in full_request['contents']:
            if content.get('role') == 'user':
                parts = content.get('parts', [])
                for part in parts:
                    text = part.get('text', '')
                    # Filter out injected context messages
                    if text and not text.strip().startswith("For context:"):
                        user_messages.append(text.strip())
        
        if not user_messages:
            return "No user text found"
            
        # Helper to clean a single message
        def clean_message(msg):
            if "</Context>" in msg:
                msg = msg.split("</Context>")[-1].strip()
            elif "</context>" in msg:
                msg = msg.split("</context>")[-1].strip()
            
            if len(msg) > 500:
                lower_msg = msg.lower()
                if "for the question" in lower_msg:
                    idx = lower_msg.find("for the question")
                    msg = msg[idx:].strip()
            return msg

        # Get the last meaningful message and clean it
        last_message = clean_message(user_messages[-1])
        
        # If result is short and there was history, join with previous message
        if len(user_messages) > 1 and len(last_message) < 50:
             prev_message = user_messages[-2]
             # Don't aggressively clean prev message, just basic check
             if not prev_message.strip().startswith("For context:"):
                 combined = f"{prev_message} | {last_message}"
                 return combined[:150]
             
        return last_message[:150]
            
    except Exception as e:
        return f"Error extracting query: {str(e)}"
