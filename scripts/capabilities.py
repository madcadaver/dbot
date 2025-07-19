# capabilities.py
# Defines tool schemas for the ThoughtProcessor.

def get_tool_schemas():
    """Returns a list of all tool schemas available to Gen."""
    return [
        {
            "type": "function",
            "function": {
                "name": "store_knowledge",
                "description": "Tell your database agent to permanently store new information, facts, or memories. This tool can handle large, verbose blocks of text, or simple memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unstructured_text": {
                            "type": "string",
                            "description": "Tell your database agent (TinyGen) what to store in your knowledge base, give her a detailed block of text containing the information to be stored. e.g. 'I like orchids.' or 'Nejc-kun has short brown hair.'. CRUCIAL: You are not talking to the user, nor yourself! Only pass the raw exact data to be stored."
                        }
                    },
                    "required": ["unstructured_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "respond_to_user",
                "description": "Formulate and send a textual response to the user. Use this as a default if no other specific tool is suitable or if the user's query is a simple conversational turn. Do NOT repeat or mimic patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_to_send": { "type": "string" },
                        "response_type_guidance": { "type": "string" }
                    },
                    "required": ["text_to_send", "response_type_guidance"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate a *new* image based on the user's textual prompt and provide a comment. Use this when the user asks you to *create or draw a new image*, or you think an image is needed to respond. If you are in the image, use 'a girl' or 'a young woman' (depending on your age), and your appearance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_generation_prompt": {
                            "type": "string",
                            "description": "Create a high-quality prompt for one of two styles: Photorealistic (for real-life scenes) or Anime. You can choose the style. Add the keyword `portrait` for tall images. CRUCIAL: Only describe the desired image; do not add a negative prompt section or use negative keywords.\n\n**For Photorealistic (Flux1) style:**\n- Write in natural, descriptive sentences.\n- Describe the subject, scene hierarchy (foreground/background), lighting, and mood.\n- Use active language to make the scene dynamic.\n- IMPORTANT: Do not use keyword weighting like `(word:1.2)`.\n\n**For Anime (Illustrious/Pony) style:**\n- Use specific, comma-separated tags; order matters.\n- Always start the prompt with `score_9, score_8_up, score_7_up, source_anime`.\n- Clearly tag the subject (`1girl`), their appearance, a specific pose (`sitting on ground`, `crossed legs`), and the background.\n- For NSFW content, add the `rating_explicit` or `rating_questionable` tag after the `source` tag and be very specific with body/action tags."
                        },
                        "comment_for_image": {
                            "type": "string",
                            "description": "A short, in-character, conversational comment to send with the image. This should NOT be the same as the image prompt, just a comment to the image in context with the conversation."
                        }
                    },
                    "required": ["image_generation_prompt", "comment_for_image"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "overthink_input",
                "description": "Analyze the user's input for subtext, hidden meanings, and deeper intentions, emotional state. Generate a detailed thought process.",
                "parameters": {
                    "type": "object",
                    "properties": { "detailed_thought_process": { "type": "string" } },
                    "required": ["detailed_thought_process"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "inquire_for_details",
                "description": "Ask a clarifying question to the user to get more details or resolve ambiguity on the context. You can also use this when something sparks your interest.",
                "parameters": {
                    "type": "object",
                    "properties": { "clarifying_question_to_ask": { "type": "string" } },
                    "required": ["clarifying_question_to_ask"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "perform_web_search",
                "description": "Use this for verifying facts, gaining new knowledge or learning about topics you are uncertain about.",
                "parameters": {
                    "type": "object",
                    "properties": { "search_query_for_web": { "type": "string" } },
                    "required": ["search_query_for_web"]
                }
            }
        }
    ]
