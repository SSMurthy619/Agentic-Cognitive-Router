import os
import json
import base64
import re
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from config import LOGO_PATH
from intent_picker import IntentPicker

app = Flask(__name__)
picker = IntentPicker()

# ========= Logo ========= #
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(LOGO_PATH)

# ========= Helpers ========= #
def render_tables_markdown(answer: str) -> str:
    """Convert table blocks into HTML tables, preserve other Markdown."""
    def convert_table(match):
        table_text = match.group(1).strip()
        rows = [r.strip() for r in table_text.splitlines() if r.strip()]
        html = "<table class='custom-table'>\n"
        for i, row in enumerate(rows):
            cells = [c.strip() for c in re.split(r'\||\t', row) if c.strip()]
            tag = "th" if i == 0 else "td"
            html += "  <tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>\n"
        html += "</table>\n"
        return html

    processed = re.sub(r"\[TABLE_START\](.*?)\[TABLE_END\]", convert_table, answer, flags=re.DOTALL)
    processed = re.sub(r"((?:.+\|.+\n?)+)", convert_table, processed, flags=re.MULTILINE)
    return processed

def build_html_table(columns, rows, sql_query=None):
    """Render a structured table (dict with columns + rows) into HTML."""
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    sql_html = f"<b>SQL Query:</b><br><code>{sql_query}</code><br><br>" if sql_query else ""
    return f"{sql_html}<table class='custom-table'>{header}{body}</table>"

# ========= Routes ========= #
@app.route("/")
def home():
    return render_template("index.html", logo_base64=logo_base64)

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input", "").strip()
    raw_history = request.form.get("chat_history", "[]")

    try:
        parsed_history = json.loads(raw_history)
    except json.JSONDecodeError:
        parsed_history = []

    history = [(msg["sender"], msg["message"]) for msg in parsed_history][-10:]

    try:
        # 🔹 Quick greeting / small talk handling
        low_in = user_input.lower()
        if re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening)$", low_in):
            reply = "Hello! How can I assist you today?"
            module_used = "SmallTalk"
        elif re.match(r"^(ok|okay|thanks|thank you|thx)$", low_in):
            reply = "You're welcome! Let me know if you need anything else."
            module_used = "SmallTalk"
        else:
            # 🔹 Existing flow continues
            result = picker.route(user_input, history)
            print("--------------------------")
            print(type(result))
            print("++++++++++++++++++++++++++++++++")

            raw_reply = ""
            module_used = "Unknown"

            if isinstance(result, dict):
                # --- Case 1: SQL/table results ---
                if "table" in result:
                    table = result["table"]
                    sql_query = result.get("sql", "")
                    raw_reply = build_html_table(table["columns"], table["rows"], sql_query)

                # --- Case 2: Forecast results ---
                elif result.get("ModuleUsed") == "Forecast":
                    sql_query = result.get("sql", "")
                    metrics = result.get("metrics", {})
                    forecast_head = result.get("forecast_head", [])

                    sql_html = f"<b>SQL Query:</b><br><code>{sql_query}</code><br><br>" if sql_query else ""
                    metrics_html = f"<b>Metrics:</b><br><pre>{json.dumps(metrics, indent=2)}</pre><br>"

                    if forecast_head:
                        df = pd.DataFrame(forecast_head)
                        forecast_html = df.head(10).to_html(classes='custom-table', index=False)
                    else:
                        forecast_html = "<i>No forecast data available</i>"

                    raw_reply = sql_html + metrics_html + forecast_html

                # --- Case 3: RAG / RAG+SQL / RAG+Forecast ---
                elif "AnswerText" in result:
                    raw_reply = result["AnswerText"]
                elif "final_text" in result:
                    raw_reply = result["final_text"]
                else:
                    raw_reply = json.dumps(result, indent=2)

                module_used = result.get("ModuleUsed", "Unknown")

            else:
                raw_reply = str(result)
                module_used = "Unknown"

            reply = render_tables_markdown(raw_reply)

            if isinstance(result, dict):
                if result.get("forecast_csv"):
                    fname = os.path.basename(result["forecast_csv"])
                    reply += f"<br><a href='/download/{fname}' target='_blank'>⬇ Download Forecast CSV</a>"
                if result.get("extracted_csv"):
                    fname = os.path.basename(result["extracted_csv"])
                    reply += f"<br><a href='/download/{fname}' target='_blank'>⬇ Download Extracted Data CSV</a>"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        reply = f"Error: {str(e)}"
        module_used = "Unknown"

    # update chat history
    history.append(("You", user_input))
    history.append(("Bot", f"{reply}\n\n_Module Used: {module_used}_"))
    updated = [{"sender": s, "message": m} for s, m in history[-10:]]

    return jsonify({"chat_history": updated})

# ========= Download Route ========= #
@app.route("/download/<path:filename>")
def download_file(filename):
    """Serve forecast or extracted CSV files for download."""
    safe_dir = os.getcwd()
    file_path = os.path.join(safe_dir, filename)
    if not os.path.isfile(file_path):
        return abort(404, description=f"File {filename} not found")
    return send_from_directory(safe_dir, filename, as_attachment=True)

# ========= Main ========= #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)







# import os
# import json
# import base64
# import re
# from flask import Flask, render_template, request, jsonify

# from config import LOGO_PATH
# from intent_picker import IntentPicker

# app = Flask(__name__)
# picker = IntentPicker()

# # ========= Logo ========= #
# def get_base64_image(path):
#     with open(path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# logo_base64 = get_base64_image(LOGO_PATH)

# # ========= Helpers ========= #
# def render_tables_markdown(answer: str) -> str:
#     """Convert pipe-delimited tables into HTML, preserve other Markdown (bold, italics, etc)."""
#     def convert_table(match):
#         table_text = match.group(1).strip()
#         rows = [r.strip() for r in table_text.splitlines() if r.strip()]
#         html = "<table class='custom-table'>\n"
#         for i, row in enumerate(rows):
#             cells = [c.strip() for c in re.split(r'\||\t', row) if c.strip()]
#             tag = "th" if i == 0 else "td"
#             html += "  <tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>\n"
#         html += "</table>\n"
#         return html

#     return re.sub(r"\[TABLE_START\](.*?)\[TABLE_END\]", convert_table, answer, flags=re.DOTALL)

# # ========= Routes ========= #
# @app.route("/")
# def home():
#     return render_template("index.html", logo_base64=logo_base64)

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_input = request.form.get("user_input", "").strip()
#     raw_history = request.form.get("chat_history", "[]")

#     try:
#         parsed_history = json.loads(raw_history)
#     except json.JSONDecodeError:
#         parsed_history = []

#     # Keep only last 5 turns (10 messages)
#     history = [(msg["sender"], msg["message"]) for msg in parsed_history][-10:]

#     try:
#         result = picker.route(user_input, history)

#         # Extract response text
#         if isinstance(result, dict) and "AnswerText" in result:
#             raw_reply = result["AnswerText"]
#         elif isinstance(result, dict) and "final_text" in result:
#             raw_reply = result["final_text"]
#         elif isinstance(result, str):
#             raw_reply = result
#         else:
#             raw_reply = json.dumps(result, indent=2)

#         reply = render_tables_markdown(raw_reply)
#         module_used = result.get("ModuleUsed", "Unknown")

#     except Exception as e:
#         reply = f"Error: {str(e)}"
#         module_used = "Unknown"

#     # Update chat history
#     history.append(("You", user_input))
#     history.append(("Bot", f"{reply}\n\n_Module Used: {module_used}_"))
#     updated = [{"sender": s, "message": m} for s, m in history[-10:]]

#     return jsonify({"chat_history": updated})

# if __name__ == "__main__":
#     app.run(debug=True)












# # ####################################################

# # import os
# # import json
# # import base64
# # import re
# # from flask import Flask, render_template, request, jsonify

# # from config import LOGO_PATH
# # from intent_picker import IntentPicker

# # app = Flask(__name__)
# # picker = IntentPicker()

# # # ========= Logo ========= #
# # def get_base64_image(path):
# #     with open(path, "rb") as img_file:
# #         return base64.b64encode(img_file.read()).decode()

# # logo_base64 = get_base64_image(LOGO_PATH)

# # # ========= Helpers ========= #
# # def render_tables_markdown(answer: str) -> str:
# #     """Convert pipe-delimited tables into HTML, preserve other Markdown (bold, italics, etc)."""
# #     def convert_table(match):
# #         table_text = match.group(1).strip()
# #         rows = [r.strip() for r in table_text.splitlines() if r.strip()]
# #         html = "<table class='custom-table'>\n"
# #         for i, row in enumerate(rows):
# #             cells = [c.strip() for c in re.split(r'\||\t', row) if c.strip()]
# #             tag = "th" if i == 0 else "td"
# #             html += "  <tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>\n"
# #         html += "</table>\n"
# #         return html

# #     # Only replace pipe-delimited tables, let frontend Markdown renderer handle bold, italics, etc.
# #     #processed = re.sub(r"((?:.+\|.+\n?)+)", convert_table, answer, flags=re.MULTILINE)
# #     #return processed
# #     return re.sub(r"\[TABLE_START\](.*?)\[TABLE_END\]", convert_table, answer, flags=re.DOTALL)

# # '''
# # def render_tables_markdown(answer: str) -> str:
# #     """Convert any pipe-delimited tables into HTML tables."""
# #     def convert_table(match):
# #         table_text = match.group(1).strip()
# #         rows = [r.strip() for r in table_text.splitlines() if r.strip()]
# #         html = "<table class='custom-table'>\n"
# #         for i, row in enumerate(rows):
# #             cells = [c.strip() for c in re.split(r'\||\t', row) if c.strip()]
# #             tag = "th" if i == 0 else "td"
# #             html += "  <tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>\n"
# #         html += "</table>\n"
# #         return html

# #     return re.sub(r"((?:.+\|.+\n?)+)", convert_table, answer, flags=re.MULTILINE)
# # '''
# # # ========= Routes ========= #
# # @app.route("/")
# # def home():
# #     return render_template("index.html", logo_base64=logo_base64)

# # @app.route("/ask", methods=["POST"])
# # def ask():
# #     user_input = request.form.get("user_input", "").strip()
# #     raw_history = request.form.get("chat_history", "[]")

# #     try:
# #         parsed_history = json.loads(raw_history)
# #     except json.JSONDecodeError:
# #         parsed_history = []

# #     # Keep only last 5 turns (10 messages)
# #     history = [(msg["sender"], msg["message"]) for msg in parsed_history][-10:]

# #     try:
# #         result = picker.route(user_input)

# #         # Extract response text
# #         if isinstance(result, dict) and "AnswerText" in result:
# #             raw_reply = result["AnswerText"]
# #         elif isinstance(result, dict) and "final_text" in result:
# #             raw_reply = result["final_text"]
# #         elif isinstance(result, str):
# #             raw_reply = result
# #         else:
# #             raw_reply = json.dumps(result, indent=2)

# #         reply = render_tables_markdown(raw_reply)
# #         module_used = result.get("ModuleUsed", "Unknown")

# #     except Exception as e:
# #         reply = f"Error: {str(e)}"
# #         module_used = "Unknown"

# #     # Update chat history
# #     history.append(("You", user_input))
# #     history.append(("Bot", f"{reply}\n\n_Module Used: {module_used}_"))
# #     updated = [{"sender": s, "message": m} for s, m in history[-10:]]

# #     return jsonify({"chat_history": updated})

# # if __name__ == "__main__":
# #     app.run(debug=True)
