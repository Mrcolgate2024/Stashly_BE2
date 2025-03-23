import re
from langchain_core.messages import AIMessage, HumanMessage
from agents.research_agent.analysts import create_analysts
from agents.research_agent.interview import (
    generate_question,
    search_context,
    generate_answer,
    route_messages,
    save_interview,
    write_section
)
from agents.research_agent.report_writer import (
    write_intro_or_conclusion,
    write_report,
    finalize_report
)

def extract_topic_and_count(message: str):
    lowered = message.lower()
    count_match = re.search(r"(\d+)[- ]?(analyst|person)", lowered)
    count = int(count_match.group(1)) if count_match else 3
    topic = message.replace("research", "").replace("study", "").replace("analyze", "").strip()
    topic = re.sub(r"(with|using|by)? ?\d+[- ]?(analyst|person).*", "", topic, flags=re.IGNORECASE)
    return topic.strip().capitalize(), min(max(count, 1), 10)

def run_research_agent(state, settings=None):
    try:
        user_msg = state.get("messages", [])[-1].content
        topic, num_analysts = extract_topic_and_count(user_msg)
        feedback = ""

        # Step 1: Create analysts
        analyst_state = create_analysts({
            "topic": topic,
            "max_analysts": num_analysts,
            "human_analyst_feedback": feedback,
            "analysts": []
        })
        analysts = analyst_state["analysts"]

        # Step 2: Interview each in parallel
        interview_sections = []
        for analyst in analysts:
            interview_state = {
                "analyst": analyst,
                "messages": [HumanMessage(content=f"So you're researching {topic}?")],
                "max_num_turns": 2,
                "context": [],
                "interview": "",
                "sections": []
            }

            while True:
                q = generate_question(interview_state)
                interview_state["messages"] += q["messages"]
                ctx = search_context(interview_state)
                interview_state["context"] += ctx["context"]
                a = generate_answer(interview_state)
                interview_state["messages"] += a["messages"]
                route = route_messages(interview_state)
                if route == "save_interview":
                    break

            saved = save_interview(interview_state)
            interview_state["interview"] = saved["interview"]
            section = write_section(interview_state)
            interview_sections += section["sections"]

        # Step 3: Compile report
        intro = write_intro_or_conclusion(interview_sections, topic, "introduction")
        summary = write_report(interview_sections, topic)
        conclusion = write_intro_or_conclusion(interview_sections, topic, "conclusion")
        report = finalize_report(intro, summary, conclusion)

        return {
            "messages": [AIMessage(content=f"🧾 Final Research Report on **{topic}**:\n\n{report}")]
        }

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Research agent error: {str(e)}")]
        }
