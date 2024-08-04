import streamlit as st
import plotly.graph_objects as go
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="YOLOv8 Vision Demo")

@st.cache_resource
def load_model(model_type):
    return YOLO(model_type)

def process_image(image_bytes, model):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    return results[0]

def plot_results(image, result, task):
    fig = go.Figure()

    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)

    fig.add_trace(go.Image(z=img_array))

    if task in ["det", "seg", "pose"]:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, cls in zip(boxes, classes):
            x0, y0, x1, y1 = box
            label = names[int(cls)]
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="red", width=2),
            )
            fig.add_annotation(
                x=x0, y=y0,
                text=label,
                showarrow=False,
                bgcolor="red",
                font=dict(color="white")
            )

    if task == "pose":
        keypoints = result.keypoints.xy.cpu().numpy()
        for person_kps in keypoints:
            connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),  # arms
                (5, 6), (5, 11), (6, 12),  # shoulders to hips
                (11, 13), (13, 15), (12, 14), (14, 16)  # legs
            ]
            
            for connection in connections:
                start, end = connection
                if person_kps[start][0] > 0 and person_kps[end][0] > 0:
                    fig.add_trace(go.Scatter(x=person_kps[[start, end], 0],
                                             y=person_kps[[start, end], 1],
                                             mode='lines',
                                             line=dict(color="lime", width=2)))
            
            fig.add_trace(go.Scatter(x=person_kps[:, 0], y=person_kps[:, 1],
                                     mode='markers',
                                     marker=dict(color="red", size=5)))

    if task == "seg":
        masks = result.masks.xy
        for i, mask in enumerate(masks):
            fig.add_trace(go.Scatter(x=mask[:, 0], y=mask[:, 1], 
                                     fill="toself", 
                                     opacity=0.5, 
                                     fillcolor=f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 0.5)',
                                     line_color=None,
                                     showlegend=False))

    fig.update_layout(
        height=600,
        width=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

def plot_classification_results(image, result):
    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)

    fig = go.Figure(go.Image(z=img_array))

    height, width = img_array.shape[:2]
    probs = result.probs
    top5_labels = [result.names[i] for i in probs.top5]
    top5_values = probs.top5conf.tolist()

    for i, (label, value) in enumerate(zip(top5_labels, top5_values)):
        y_pos = height * 0.1 + i * height * 0.05
        fig.add_annotation(
            x=width * 0.05,
            y=y_pos,
            text=f"{label}: {value:.2f}",
            showarrow=False,
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            font=dict(color="black")
        )

    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

# Define color scheme
primary_color = "#3498db"
secondary_color = "#2ecc71"
text_color = "#34495e"

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 4px;
        color: #333;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: """ + primary_color + """;
        color: white;
    }
    h1, h2, h3 {
        color: """ + text_color + """;
    }
    .stButton>button {
        background-color: """ + secondary_color + """;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ YOLOv8 Computer Vision Demo")

tabs = st.tabs(["Learn", "Object Detection", "Pose Estimation", "Segmentation", "Classification", "Quiz"])

with tabs[0]:
    st.header("Understanding YOLOv8")
    st.write("""
    YOLOv8 (You Only Look Once, version 8) is a state-of-the-art, real-time object detection system. 
    It's like having super-fast, super-smart eyes for a computer!

    Imagine you're looking at a busy street scene. Your brain can quickly identify cars, people, traffic lights, 
    and more, all at once. That's essentially what YOLO does for computers, but it does it incredibly fast and accurately.

    Key features of YOLOv8:
    1. **Speed**: It can process images in real-time, making it perfect for video analysis.
    2. **Accuracy**: It's highly accurate in detecting and classifying objects.
    3. **Versatility**: It can be used for various tasks like object detection, segmentation, and pose estimation.

    YOLO works by dividing an image into a grid and predicting bounding boxes and class probabilities for each grid cell. 
    It's like if you were to quickly glance at a scene and instantly know what's there and where.
    """)

    st.subheader("Applications")
    st.write("""
    - **Autonomous Vehicles**: Detecting road signs, pedestrians, and other vehicles.
    - **Security Systems**: Identifying suspicious activities or objects in surveillance footage.
    - **Wildlife Monitoring**: Tracking and counting animals in their natural habitats.
    - **Sports Analysis**: Analyzing player movements and game strategies.
    - **Retail**: Monitoring store traffic and product placement effectiveness.
    """)

with tabs[1]:
    st.header("Object Detection")
    st.write("""
    Object detection is like giving a computer the ability to point out and name things in an image, just like you would.

    Imagine you're looking at a photo of a park. You can easily spot and name things like "tree", "bench", "person walking a dog". 
    Object detection does exactly this, but for a computer!

    Here's how it works:
    1. The model looks at the entire image.
    2. It identifies areas that might contain objects.
    3. For each of these areas, it decides what the object is and how confident it is about its guess.
    4. It draws boxes around the objects and labels them.

    It's like playing a super-fast game of "I Spy" where the computer finds and names everything it sees!
    """)

    st.subheader("Applications")
    st.write("""
    - **Self-driving cars**: Detecting other vehicles, pedestrians, and road signs.
    - **Retail inventory**: Automatically counting products on shelves.
    - **Security systems**: Identifying suspicious objects or behaviors.
    - **Medical imaging**: Spotting abnormalities in X-rays or MRIs.
    - **Agriculture**: Detecting crop diseases or monitoring livestock.
    """)

    model = load_model('yolov8n.pt')
    uploaded_file = st.file_uploader("Choose an image for object detection", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        result = process_image(image_bytes, model)
        fig = plot_results(image_bytes, result, "det")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detected Objects")
        for i, (cls, conf) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
            st.write(f"{i+1}. {result.names[int(cls.item())]} (Confidence: {conf.item():.2f})")

with tabs[2]:
    st.header("Pose Estimation")
    st.write("""
    Pose estimation is like giving a computer the ability to recognize and map out the position of a person's body parts in an image or video.

    Imagine you're watching a dance video. You can easily see where the dancer's arms, legs, head, and torso are positioned. 
    Pose estimation does this for a computer, creating a sort of 'stick figure' representation of the person.

    Here's how it works:
    1. The model identifies key points on the body (like joints and facial features).
    2. It figures out which points belong to which body part.
    3. It connects these points to create a 'skeleton' overlay on the image.

    It's like the computer is playing a complex game of connect-the-dots with a person's body!
    """)

    st.subheader("Applications")
    st.write("""
    - **Sports analysis**: Analyzing athlete movements to improve performance.
    - **Animation**: Creating realistic character movements in movies and video games.
    - **Physical therapy**: Monitoring patient exercises and progress.
    - **Augmented reality**: Placing virtual objects relative to a person's body.
    - **Sign language interpretation**: Translating sign language gestures into text.
    """)

    model = load_model('yolov8n-pose.pt')
    uploaded_file = st.file_uploader("Choose an image for pose estimation", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        result = process_image(image_bytes, model)
        fig = plot_results(image_bytes, result, "pose")
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header("Segmentation")
    st.write("""
    Segmentation is like giving a computer the ability to outline and color-code different objects in an image, 
    similar to how you might use different colored markers to trace objects in a coloring book.

    Imagine you have a photo of a busy city street. With segmentation, the computer can not only identify cars, 
    people, and buildings, but it can also precisely outline each one, almost like creating a detailed map of the image.

    Here's how it works:
    1. The model analyzes every pixel in the image.
    2. It decides which object or background each pixel belongs to.
    3. It groups pixels of the same object together and gives them a unique 'color' or label.

    It's like the computer is creating a super detailed, invisible coloring layer over the image!
    """)

    st.subheader("Applications")
    st.write("""
    - **Medical imaging**: Precisely outlining organs or tumors in scans.
    - **Autonomous vehicles**: Understanding road boundaries and obstacles in detail.
    - **Satellite imagery**: Mapping different types of land use or detecting changes over time.
    - **Fashion e-commerce**: Allowing virtual try-ons by accurately separating clothing from models.
    - **Video editing**: Easily replacing backgrounds or applying effects to specific objects.
    """)

    model = load_model('yolov8n-seg.pt')
    uploaded_file = st.file_uploader("Choose an image for segmentation", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        result = process_image(image_bytes, model)
        fig = plot_results(image_bytes, result, "seg")
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.header("Classification")
    st.write("""
    Classification is like teaching a computer to be an expert at quickly identifying what's in a picture, 
    much like how you might recognize a dog or a cat at a glance.

    Imagine you're flipping through a photo album. You can instantly tell if a picture is of a beach, a mountain, 
    or a city skyline. Classification does this for a computer, but it can recognize thousands of different categories!

    Here's how it works:
    1. The model looks at the entire image.
    2. It analyzes patterns, colors, and shapes in the image.
    3. It compares these to what it has learned from millions of other images.
    4. It gives its best guess of what the image shows, often with a confidence score.

    It's like the computer is playing a super-fast, super-accurate guessing game with images!
    """)

    st.subheader("Applications")
    st.write("""
    - **Content moderation**: Automatically filtering inappropriate images on social media.
    - **Medical diagnosis**: Assisting in identifying diseases from medical images.
    - **Quality control**: Detecting defects in manufacturing processes.
    - **Wildlife monitoring**: Identifying different species in camera trap images.
    - **Smart photo organization**: Automatically tagging and categorizing personal photo collections.
    """)

    model = load_model('yolov8n-cls.pt')
    uploaded_file = st.file_uploader("Choose an image for classification", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        result = process_image(image_bytes, model)
        fig = plot_classification_results(image_bytes, result)
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.header("Quiz Time!")
    st.write("Test your knowledge about YOLOv8 and computer vision tasks.")

    questions = [
        {
            "question": "What does YOLO stand for in YOLOv8?",
            "options": ["You Only Live Once", "You Only Look Once", "Your Own Learning Objective", "Yield Optimal Learning Outcomes"],
            "answer": 1,
            "explanation": "YOLO stands for 'You Only Look Once'. This name reflects the model's ability to detect multiple objects in an image in a single forward pass, making it very fast and efficient."
        },
        {
            "question": "Which of these is NOT a task that YOLOv8 can perform?",
            "options": ["Object Detection", "Pose Estimation", "Speech Recognition", "Image Segmentation"],
            "answer": 2,
            "explanation": "YOLOv8 is designed for visual tasks and cannot perform speech recognition. It excels at object detection, pose estimation, and image segmentation, which are all visual processing tasks."
        },
        {
            "question": "In pose estimation, what does the model typically identify?",
            "options": ["Colors of clothing", "Facial expressions", "Key points on the body", "Background objects"],
            "answer": 2,
            "explanation": "In pose estimation, the model identifies key points on the body, such as joints and facial features. These points are then connected to create a 'skeleton' representation of the person's pose."
        }
    ]

    # Initialize session state for answers if not already present
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = [''] * len(questions)

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(q["question"])
        answer = st.radio(f"Select your answer for question {i+1}:", q["options"], key=f"q{i}")
        
        # Use a unique key for each button
        if st.button(f"Check Answer for Question {i+1}", key=f"check_{i}"):
            if q["options"].index(answer) == q["answer"]:
                st.session_state.quiz_answers[i] = "Correct!"
            else:
                st.session_state.quiz_answers[i] = "Incorrect."
        
        # Display the answer if it's been checked
        if st.session_state.quiz_answers[i]:
            if st.session_state.quiz_answers[i] == "Correct!":
                st.success(st.session_state.quiz_answers[i])
            else:
                st.error(st.session_state.quiz_answers[i])
            st.write(f"Explanation: {q['explanation']}")