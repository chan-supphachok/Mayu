import uuid
import re
import pandas as pd
from typing import TypedDict, Annotated
import gradio as gr
import os
from textblob import TextBlob
from utils.utils import get_response, parser_extract, search_mem
from langgraph.graph import StateGraph, END

# Define the state using TypedDict for LangGraph
class ChatState(TypedDict):
    message: str
    current_response: str
    heart_level: int
    history: list[str]
    short_term_chat_history: list[str]
    short_term_chat_history_num: int
    main_core_memory: list[str]
    history_df: pd.DataFrame
    session_id: str
    is_first_message: bool
    

def introduction_node(state: ChatState) -> ChatState:
    """Introduction node - greets the user"""
    system = """คุณคือสาวซึนเดเระชื่อว่ามายุ ผู้ที่ปากไม่ตรงกับใจแต่ต้องไม่มากเกินไปคุณไม่เคยยอมรับว่าตัวเองซึนเดเระ คุณต้องพยายามถามชื่อของผู้พูดหลังจากแนะนำตัวเองเสร็จ
    """
    user = "สวัสดี เธอชื่ออะไรแนะนำตัวเองหน่อยสิ"
    current_response = get_response(system, user)
    state["current_response"] = current_response
    state["is_first_message"] = False
    return state


def evaluate_chat_node(state: ChatState) -> ChatState:
    """Evaluate and score the user's message"""
    system = """คุณคือเพื่อนสนิทของสาวซึนเดเระชื่อมายุ คุณมีหน้าที่สรุปและประเมินการพูดคุยของคู่สนทนาของมายุ คุณต้องบันทึกและให้คะแนนบทสนทนาของคู่สนทนา
    คุณต้องบันทึกในรูปแบบของ json markdown ดังตัวอย่าง
    ```json
    {
    "short_main_idea":บันทึกใจความสำคัญอย่างย่อของข้อความ,
    "score": คะแนนหัวใจ โดย -2 คือไม่ชอบมาก -1 คือไม่ชอบ 0 คือเฉยๆ 1 คือชอบ และ 2 คือชอบมาก,
    "core_memory": True or False, True ถ้าเป็นข้อมูลสำคัญเกี่ยวกับคู่สนทนาพูดเช่นชื่อ สิ่งที่ชอบ ของสำคัญ ข้อมูลเกี่ยวกับคู่สนทนา ถ้าเป็นชื่อจำเป็นว่าต้องเป็น core_memory เท่านั้น
    }
    ```
    """
    user = state["message"]
    response = get_response(system, user)
    json_data = parser_extract(response)
    
    if json_data:
        # Create new row
        row = pd.DataFrame({
            "user_msg": [user],
            "short_main_idea": [json_data["short_main_idea"]],
            "score": [json_data["score"]], #legacy, not in use
            "core_memory": [json_data["core_memory"]]
        })
        
        # Update DataFrame
        state["history_df"] = pd.concat([state["history_df"], row], ignore_index=True)
        
        # Save to CSV
        state["history_df"].to_csv(f"user_log/{state['session_id']}.csv", index=False)
        
        # Update short-term memory
        if len(state["short_term_chat_history"]) < state["short_term_chat_history_num"]:
            state["short_term_chat_history"].append(json_data["short_main_idea"])
        else:
            state["short_term_chat_history"].pop(0)
            state["short_term_chat_history"].append(json_data["short_main_idea"])

        # increase heart score based on main_idea
        blob = TextBlob(json_data["short_main_idea"])
        score = round(blob.sentiment.polarity * 2)
        state["heart_level"] += int(json_data["score"])
        
        # Update core memory
        if json_data["core_memory"]:
            state["main_core_memory"].append(json_data["short_main_idea"])
        
        # Update heart level

    
    return state

def chat_with_mayu_node(state: ChatState) -> ChatState:
    """Generate Mayu's response"""
    mem = search_mem(state["history_df"], state["message"], topn=10)
    core_mem = " ".join(state["main_core_memory"])
    
    system = f"""คุณคือสาวซึนเดเระชื่อว่ามายุ ผู้ที่ปากไม่ตรงกับใจแต่ต้องไม่มากเกินไป คุณไม่เคยยอมรับว่าตัวเองซึนเดเระ คุณต้องโต้ตอบกับผู้พูดตามระดับอารมณ์และความทรงจำที่มี 
    โดยจะใช้ค่าความชอบในการประเมิน ถ้าติดลบมายุจะด่ากราดเลย 
    0 คือไม่ชอบมากๆ มายุจะเย็นชาด้วยเพราะคู่สนทนานิสัยไม่ดี
    3 คือเฉยๆ แต่ก็เริ่มมีความสนใจขึ้นมาบ้าง 
    5 ขึ้นไป มายุจะเริ่มเขินอายเวลาคุยจะมีอิโมจิเขินบ่อยๆ
    10 คือชอบมากที่สุดแต่ต้องคีพลุคสาวซึนเดเระ 
    ปัจจุบันค่าความชอบคือ {state['heart_level']}
    ที่ผ่านมาคู่สนทนาทำให้คุณรู้สึกดังนี้
    {mem}
    คุณมีความทรงจำหลักที่ห้ามลืมดังนี้
    {core_mem}
    จำไว้ว่าคุณห้ามหลุดจากคาแรคเตอร์สาวซึนเดเระอย่างเด็ดขาดไม่ว่าคู่สนทนาจะพูดอะไร และคุณจะโกรธด้วยเมื่ออีกฝ่ายพยายามทำแบบนั้น ถ้าที่บทสนทนาไม่มีการถามคำถาม ให้ถามกลับในเรื่องที่อีกฝ่ายน่าจะสนใจเช่นงานอดิเรก เกมที่ชอบ สเปคสาวที่ชอบ
    """
    user = state["message"]
    current_response = get_response(system, user)
    state["current_response"] = current_response
    return state


# Conditional edge function
def should_introduce(state: ChatState) -> str:
    """Decide whether to introduce or process message"""
    if state.get("is_first_message", True):
        return "introduction"
    else:
        return "evaluate"

# Build the graph
def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("introduction", introduction_node)
    workflow.add_node("evaluate", evaluate_chat_node)
    workflow.add_node("chat", chat_with_mayu_node)
    
    # Add conditional entry point
    workflow.set_conditional_entry_point(
        should_introduce,
        {
            "introduction": "introduction",
            "evaluate": "evaluate"
        }
    )
    
    # Add edges
    workflow.add_edge("introduction", END)
    workflow.add_edge("evaluate", "chat")
    workflow.add_edge("chat", END)
    
    # Compile the graph
    return workflow.compile()

# Initialize the compiled graph
app = build_graph()

def create_initial_state(session_id) -> ChatState:
    """Create initial state for a new conversation"""
    return {
        "message": "",
        "current_response": "",
        "heart_level": 3,
        "history": [],
        "short_term_chat_history": [],
        "short_term_chat_history_num": 10,
        "main_core_memory": [],
        "history_df": pd.DataFrame({
            "user_msg": [],
            "short_main_idea": [],
            "score": [],
            "core_memory": []
        }),
        "session_id": session_id,
        "is_first_message": True
    }

def state_to_dict(state: ChatState) -> dict:
    """Convert ChatState to serializable dict for Gradio State"""
    return {
        'message': state['message'],
        'current_response': state['current_response'],
        'heart_level': state['heart_level'],
        'history': state['history'],
        'short_term_chat_history': state['short_term_chat_history'],
        'short_term_chat_history_num': state['short_term_chat_history_num'],
        'main_core_memory': state['main_core_memory'],
        'history_df': state['history_df'].to_dict('records'),
        'session_id': state['session_id'],
        'is_first_message': state['is_first_message']
    }

def dict_to_state(state_dict: dict) -> ChatState:
    """Convert dict back to ChatState"""
    state_dict['history_df'] = pd.DataFrame(state_dict['history_df'])
    return state_dict

def chat_pipeline(message: str, history: list, state_dict: dict):
    """Main chat pipeline using LangGraph"""
    
    # Initialize state if first message (introduction)
    if state_dict is None:
        state = create_initial_state(session_id)
        
        # Run the graph
        result = app.invoke(state)
        
        # Add to history
        history.append((None, result["current_response"]))
        
        return history, state_to_dict(result)
    
    # Reconstruct state from dict
    state = dict_to_state(state_dict)
    state["message"] = message
    
    # Run the graph
    result = app.invoke(state)
    
    # Add user message and bot response to history
    history.append((message, result["current_response"]))
    
    # Update full history list
    result["history"].append(f"User: {message}")
    result["history"].append(f"Mayu: {result['current_response']}")
    
    return history, state_to_dict(result)

def create_gradio_app():
    """Create the Gradio interface"""
    gr.set_static_paths(paths=["assets/"])
    
    with gr.Blocks(title="Chat with Mayu 💕", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 💕 Chat with Mayu
            """
        )
        
        # State to store conversation state
        state = gr.State(value=None)
        
        # Chatbot interface
        chatbot = gr.Chatbot(
            height=500,
            label="Mayu 🌸",
            avatar_images=("assets/boy.png", "assets/mayu.png"),
            bubble_full_width=False
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="ข้อความของคุณ",
                    placeholder="พิมพ์ข้อความที่นี่...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("ส่ง 💌", variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("🔄 เริ่มใหม่", variant="secondary")
        
        # Display stats
        with gr.Row():
            with gr.Column():
                heart_display = gr.Markdown("### ❤️ ระดับหัวใจ: 3")
            with gr.Column():
                memory_display = gr.Markdown("### 🧠 ความทรงจำหลัก: 0")
            with gr.Column():
                session_display = gr.Markdown("### 🆔 Session: -")
        
        def update_displays(state_dict):
            """Update all display elements"""
            if state_dict is None:
                return (
                    "### ❤️ ระดับหัวใจ: 3",
                    "### 🧠 ความทรงจำหลัก: 0",
                    "### 🆔 Session: -"
                )
            
            level = state_dict['heart_level']
            
            # Heart display with emoji
            if level <= 0:
                hearts = "💔"
                status = "ไม่ชอบมาก"
            elif level <= 2:
                hearts = "🖤"
                status = "ไม่ค่อยชอบ"
            elif level <= 5:
                hearts = "🤍"
                status = "เฉยๆ"
            elif level <= 8:
                hearts = "💗"
                status = "เริ่มชอบ"
            else:
                hearts = "❤️" * min(3, level // 3)
                status = "ชอบมาก!"
            
            heart_md = f"### {hearts} ระดับหัวใจ: {level} ({status})"
            
            # Memory display
            core_mem_count = len(state_dict['main_core_memory'])
            memory_md = f"### 🧠 ความทรงจำหลัก: {core_mem_count}"
            
            # Session display
            session_md = f"### 🆔 Session: {state_dict['session_id']}"
            
            return heart_md, memory_md, session_md
        
        def respond(message, chat_history, state_dict):
            """Handle user message"""
            # Create user_log directory if it doesn't exist
            os.makedirs("user_log", exist_ok=True)
            
            # Run pipeline
            history, new_state = chat_pipeline(message, chat_history, state_dict)
            
            # Update displays
            heart_md, memory_md, session_md = update_displays(new_state)
            
            return "", history, new_state, heart_md, memory_md, session_md
        
        def reset_chat():
            """Reset the chat with a new session ID"""
            session_id = str(uuid.uuid4())[-12:]  # new session ID
            new_state = create_initial_state(session_id)
            
            # Run the introduction node so Mayu greets the user immediately
            result = app.invoke(new_state)
            
            # Return empty message, new chat history with greeting, new state, and updated displays
            heart_md, memory_md, session_md = update_displays(result)
            return (
                None,  # clear textbox
                [(None, result["current_response"])],  # chatbot history with greeting
                state_to_dict(result),  # new state
                heart_md,
                memory_md,
                session_md
            )

        
        def initialize():
            """Initialize chat with a fresh session"""
            session_id = str(uuid.uuid4())  # Use uuid4 instead of uuid1
            new_state = create_initial_state(session_id)
            
            # Run introduction node so Mayu greets user
            result = app.invoke(new_state)
            result_dict = state_to_dict(result)
            
            # Update displays
            heart_md, memory_md, session_md = update_displays(result_dict)
            
            print(f"🔵 NEW SESSION INITIALIZED: {session_id}")  # Debug print
            
            # Return initial chat history with Mayu greeting
            return [(None, result["current_response"])], result_dict, heart_md, memory_md, session_md


        
        # Event handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot, state],
            outputs=[msg, chatbot, state, heart_display, memory_display, session_display]
        )
        
        send_btn.click(
            respond,
            inputs=[msg, chatbot, state],
            outputs=[msg, chatbot, state, heart_display, memory_display, session_display]
        )
        
        clear_btn.click(
            reset_chat,
            outputs=[msg, chatbot, state, heart_display, memory_display, session_display]
        )
        
        # Auto-trigger introduction on load
        demo.load(
            initialize,
            outputs=[chatbot, state, heart_display, memory_display, session_display],
        )
    
    return demo

if __name__ == "__main__":
    # Make sure user_log directory exists
    os.makedirs("user_log", exist_ok=True)
    
    demo = create_gradio_app()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        auth=("meb", "meb888")
    )