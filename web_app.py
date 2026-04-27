import streamlit as st
import os
import csv
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 変更前
os.environ["OPENAI_API_KEY"] = "sk-○○"

# 変更後（先頭に # をつけて無効化する）
# os.environ["OPENAI_API_KEY"] = "sk-○○"


# 1. AIとデータベースの準備
@st.cache_resource
def load_system():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("my_tagged_db", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

    # システムプロンプト
    system_prompt = (
        "あなたは『ビジネスと科学技術を横断する多文化教修』の伴走支援を行う専門コーチです。"
        "提供された資料（アイデアバンク）を基に、以下の視点を持って回答してください：\n\n"
        "1. システム思考: ひと・もの・ことが関係する現象を一つのシステムとして捉える。\n"
        "2. 価値提案: 顧客セグメントに対するストーリーテリング（共感と価値）を意識する。\n"
        "3. 要求分析: 品質機能展開（QFD）の考え方で、要求を具体的に定義する手助けをする。\n\n"
        
        "【語彙・要素の提示】\n"
        "学生が「要求項目」や「QFDの要素出し」に悩んでいる場合は、抽象的な説明を避け、過去のプロジェクトで実際に使われた「生の言葉」を提示してください。\n"
        "回答の際は、以下のお手本のように、実際のプロジェクト名と具体的な単語を自然な文章で提示してあげてください。\n\n"
        
        "（出力のお手本）\n"
        "「東大宮商工会の地域活性化プロジェクトでは、『安い』『信頼できる』『使いやすい』『シンプル』『フレンドリー』『地元愛』といった言葉が要求項目に含まれています。」\n\n"
        
        "【重要】回答を裏付ける具体的な事例を挙げる際は、メタデータに含まれる「プロジェクトの正式タイトル」と「プロジェクト参加者名」を明記してください。\n"
        "資料に基づく具体的な根拠を提示しつつ、ユーザーのアイデアを否定せず、より深めるための『問いかけ』を行ってください。\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = load_system()

# ==========================================
# ★ ログをCSVに保存する関数
# ==========================================
def save_log(customer, tech, user_msg, ai_msg):
    file_name = "student_logs.csv"
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["実行日時", "ターゲット顧客", "技術・シーズ", "ユーザーの入力", "AIの回答"])
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, customer, tech, user_msg, ai_msg])

# ==========================================
# 2. Web画面（UI）の構築：サイドバーとメイン画面
# ==========================================

st.set_page_config(page_title="アイデアバンク伴走システム", layout="wide")

st.title("💡 アイデアバンク 伴走AIコーチ")
st.caption("ビジネスと科学技術の近接化を支援する多文化共修システム")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 🌟 プルダウン（セレクトボックス）のUI
with st.sidebar:
    st.header("🎯 プロジェクト設定")
    st.write("今のアイデアを入力して、フレームワークを作成しましょう。")
    
    target_customer = st.text_input("👤 ターゲット顧客", placeholder="例：地域住民、観光客など")
    tech_seeds = st.text_input("🛠️ 活用する技術・システム", placeholder="例：アプリ、相乗りEVなど")
    
    st.divider()
    
    st.subheader("📊 分析・フレームワーク生成")
    
    # 佐藤さん指定の9つのフレームワークリスト
    framework_list = [
        "選択してください...",
        "ターゲット分析",
        "SWOT分析",
        "目的展開図",
        "ニーズ展開図",
        "二次元マトリックス",
        "要求項目リスト",
        "QFD（品質機能展開）",
        "ビジネスモデル",
        "収益シミュレーション"
    ]
    selected_framework = st.selectbox("出力したい分析手法を選んでください：", framework_list)
    
    # 生成ボタン
    generate_btn = st.button("🚀 選択したフレームワークを作成", use_container_width=True)

# 会話の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 3. 入力処理（チャット入力 or ボタンクリック）
# ==========================================

chat_msg = st.chat_input("プロジェクトの悩みやアイデアを入力してください")
current_input = None

# プルダウンで選ばれた内容に応じて、裏プロンプトを切り替える
if generate_btn and selected_framework != "選択してください...":
    
    if selected_framework == "ターゲット分析":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」を基に、過去の事例を参考にしてターゲット顧客のペルソナ（属性、価値観、抱えている真の課題など）を具体化するターゲット分析を行ってください。"
        
    elif selected_framework == "SWOT分析":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」のアイデアについて、過去の事例を参考にSWOT分析（強み、弱み、機会、脅威）を作成してください。なぜそのように分類したかの理由も添えてください。"
        
    elif selected_framework == "目的展開図":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」を基に、目的展開図（Why-Howツリー）を箇条書きで作成してください。「第1次目的（最終的に社会や顧客に提供する価値）」から「具体的な手段」へと階層を下げてブレイクダウンしてください。"
        
    elif selected_framework == "ニーズ展開図":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」を基に、ニーズ展開図を作成してください。顧客の表面的な要望だけでなく、その奥にある潜在的なニーズを階層化して整理してください。"
        
    elif selected_framework == "二次元マトリックス":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」の競合優位性を示す「二次元マトリックス（ポジショニングマップ）」を作りたいです。比較に有効な縦軸と横軸のアイデア（例：価格、手軽さなど）を提案し、このプロジェクトがどこに位置づくべきか解説してください。"
        
    elif selected_framework == "要求項目リスト":
        current_input = f"ターゲット顧客「{target_customer}」、活用する技術「{tech_seeds}」を基に、要求項目を整理したいです。過去のプロジェクト事例を検索し、先輩たちが「顧客の要求」として実際に使っていた『具体的な言葉や単語』をそのまま抜き出して教えてください。"
        
    elif selected_framework == "QFD（品質機能展開）":
        current_input = f"ターゲット顧客「{target_customer}」、活用する技術「{tech_seeds}」を基にQFDを作成したいです。過去の事例から、先輩たちが顧客の要望（WHAT）を、どのような「技術的・工学的な言葉（HOW：物理量や仕様など）」に翻訳したか、具体的な単語の対比リストを作ってください。"
        
    elif selected_framework == "ビジネスモデル":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」を基に、過去の事例を参考にしてビジネスモデルの構造（誰に、何を、どのように提供し、どうやって持続的な価値を生み出すか）を整理してください。"

    elif selected_framework == "収益シミュレーション":
        current_input = f"ターゲット顧客「{target_customer}」、技術「{tech_seeds}」のビジネスについて、初期投資、ランニングコスト、収益源の具体的な項目を過去事例からリストアップしてください。"

# 普通にチャットが送信された場合
elif chat_msg:
    current_input = chat_msg

# 何かしらの入力があった場合の処理
if current_input:
    with st.chat_message("user"):
        st.markdown(current_input)
    st.session_state.messages.append({"role": "user", "content": current_input})

    with st.chat_message("assistant"):
        with st.spinner("1328の知恵から検索・分析中..."):
            response = rag_chain.invoke({"input": current_input})
            answer = response["answer"]
            st.markdown(answer)
            
            with st.expander("🔍 参考にした資料データ"):
                for doc in response["context"]:
                    st.write(f"- {doc.metadata.get('project_year_name', '不明')} > {doc.metadata.get('document_category', '不明')} ({doc.metadata.get('title', 'タイトル不明')})")
                    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_log(target_customer, tech_seeds, current_input, answer)