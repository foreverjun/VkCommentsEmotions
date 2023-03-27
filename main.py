import re

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from vk_api import VkApi
from typing import List, Tuple


@st.cache_resource()
def load_model() -> Tuple[BertForSequenceClassification, BertTokenizer]:
    model = BertForSequenceClassification.from_pretrained(
        "Aniemore/rubert-tiny-emotion-russian-cedr-m7"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "Aniemore/rubert-tiny-emotion-russian-cedr-m7"
    )
    return model, tokenizer


def parse_link(link: str) -> Tuple[int, int, bool]:
    pattern = re.compile(
        r"https://vk\.com/[A-Za-z]+\?w=wall-[0-9]+_[0-9]+", re.IGNORECASE
    )
    if pattern.match(link):
        ids: List[str] = link.split("w=wall", 1)[1].split("_")
        return int(ids[0]), int(ids[1]), True
    else:
        return 0, 0, False


def get_comments(
    owner_id: int, post_id: int, comments_num: int, vk: VkApi
) -> List[dict]:
    comments: List[dict] = []
    i: int = 0
    prev: int = -1
    while i < 35 and len(comments) < comments_num and len(comments) != prev:
        current_num: int = min(comments_num - len(comments), 100)
        prev: int = len(comments)
        try:
            comments_buf = vk.wall.getComments(
                owner_id=owner_id,
                post_id=post_id,
                count=current_num,
                offset=i * 100,
                thread_items_count=10,
            )["items"]
        except:
            st.warning("Не удалось получить комментарии поста")
        comments += comments_buf
        for comment in comments_buf:
            if len(comment["thread"]["items"]) + len(comments) < comments_num:
                comments += comment["thread"]["items"]
        i += 1
    return comments


def main() -> None:
    session = VkApi(token=st.secrets["VK_TOKEN"])
    vk: VkApi = session.get_api()
    st.title(
        "Определите эмоциональный окрас комментариев под постом сообщества Вконтакте"
    )
    st.write("Простое приложение для оценки эмоционального окраса комментариев")

    emoji_dict: dict = {
        "neutral": "😐",
        "anger": "😠",
        "enthusiasm": "🤩",
        "sadness": "😔",
        "fear": "😨",
        "happiness": "😊",
        "disgust": "🤢",
    }

    model, tokenizer = load_model()

    user_input: str = st.text_input(
        "Введите ссылку на пост сообщества Вконтакте, чтобы оценить их эмоциональный окрас:"
    )
    comments_num: int = st.number_input(
        "Введите максимально возможное количество комментариев (максимум 3500, учитываются первые 10 вложенных "
        "комментариев из каждой ветки)",
        min_value=0,
        max_value=3500,
        step=1,
    )

    if st.button("Оценить"):
        emotions = {
            "neutral": 0,
            "anger": 0,
            "enthusiasm": 0,
            "sadness": 0,
            "fear": 0,
            "happiness": 0,
            "disgust": 0,
        }
        owner_id, post_id, is_ok = parse_link(user_input)
        if is_ok:
            comments = get_comments(int(owner_id), int(post_id), comments_num, vk)

            for comment in comments:
                input_ids = tokenizer.encode(comment["text"], return_tensors="pt")

                with torch.no_grad():
                    logits = model(input_ids).logits
                predicted_class_id = logits.argmax().item()
                emotions[model.config.id2label[predicted_class_id]] += 1

            for emotion, count in emotions.items():
                emoji = emoji_dict.get(emotion)
                st.write(f"{emoji} {count}")
            st.write(f"Всего комментариев: {sum(emotions.values())}")

        else:
            st.warning(
                "Вы должны ввести ссылку на пост сообщества Вконтакте в подобном формате: https://vk.com/groupname?w=wall-72378974_8296684"
            )


if __name__ == "__main__":
    main()
