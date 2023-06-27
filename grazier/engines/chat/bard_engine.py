"""
Some of the code below is based on https://github.com/acheong08/Bard
Copyright (c) 2023 Antonio Cheong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import asyncio
import json
import logging
import os
import random
import re
import string
from typing import Any, Dict, List, Optional

import httpx

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.python import retry, singleton


class Chatbot:
    """
    Synchronous wrapper for the AsyncChatbot class.
    """

    def __init__(
        self,
        session_id: str,
        proxy: Optional[dict] = None,
        timeout: int = 20,
    ):
        self.loop = asyncio.get_event_loop()
        self.async_chatbot = self.loop.run_until_complete(
            AsyncChatbot.create(session_id, proxy, timeout),
        )

    def save_conversation(self, file_path: str, conversation_name: str):
        return self.loop.run_until_complete(
            self.async_chatbot.save_conversation(file_path, conversation_name),
        )

    def load_conversations(self, file_path: str) -> List[Dict]:
        return self.loop.run_until_complete(
            self.async_chatbot.load_conversations(file_path),
        )

    def load_conversation(self, file_path: str, conversation_name: str) -> bool:
        return self.loop.run_until_complete(
            self.async_chatbot.load_conversation(file_path, conversation_name),
        )

    @retry()
    def ask(self, message: str) -> dict:
        return self.loop.run_until_complete(self.async_chatbot.ask(message))


class AsyncChatbot:
    """
    A class to interact with Google Bard.
    Parameters
        session_id: str
            The __Secure-1PSID cookie.
        proxy: str
        timeout: int
            Request timeout in seconds.
    """

    __slots__ = [
        "headers",
        "_reqid",
        "SNlM0e",
        "conversation_id",
        "response_id",
        "choice_id",
        "proxy",
        "session_id",
        "session",
        "timeout",
    ]

    def __init__(
        self,
        session_id: str,
        proxy: Optional[dict] = None,
        timeout: int = 20,
    ):
        headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
        self._reqid = int("".join(random.choices(string.digits, k=4)))
        self.proxy = proxy
        self.conversation_id = ""
        self.response_id = ""
        self.choice_id = ""
        self.session_id = session_id
        self.session = httpx.AsyncClient(proxies=self.proxy)
        self.session.headers = headers
        self.session.cookies.set("__Secure-1PSID", session_id)
        self.timeout = timeout

    @classmethod
    async def create(
        cls,
        session_id: str,
        proxy: Optional[dict] = None,
        timeout: int = 20,
    ) -> "AsyncChatbot":
        instance = cls(session_id, proxy, timeout)
        instance.SNlM0e = await instance.__get_snlm0e()
        return instance

    async def save_conversation(self, file_path: str, conversation_name: str) -> None:
        """
        Saves conversation to the file
        :param file_path: file to save (json)
        :param conversation_name: any name of current conversation (unique one)
        :return: None
        """
        # Load conversations from file
        conversations = await self.load_conversations(file_path)

        # Update existing one
        conversation_exists = False
        for conversation in conversations:
            if conversation["conversation_name"] == conversation_name:
                conversation["conversation_name"] = conversation_name
                conversation["_reqid"] = self._reqid
                conversation["conversation_id"] = self.conversation_id
                conversation["response_id"] = self.response_id
                conversation["choice_id"] = self.choice_id
                conversation["SNlM0e"] = self.SNlM0e
                conversation_exists = True

        # Create conversation object
        if not conversation_exists:
            conversation = {
                "conversation_name": conversation_name,
                "_reqid": self._reqid,
                "conversation_id": self.conversation_id,
                "response_id": self.response_id,
                "choice_id": self.choice_id,
                "SNlM0e": self.SNlM0e,
            }
            conversations.append(conversation)

        # Save to the file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=4)

    async def load_conversations(self, file_path: str) -> List[Dict]:
        # Check if file exists
        if not os.path.isfile(file_path):
            return []
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    async def load_conversation(self, file_path: str, conversation_name: str) -> bool:
        """
        Loads a conversation from history file. Returns whether the conversation was found
        :param file_path: File with conversations (json)
        :param conversation_name: unique conversation name
        :return: True if the conversation was found
        """
        conversations = await self.load_conversations(file_path)
        for conversation in conversations:
            if conversation["conversation_name"] == conversation_name:
                self._reqid = conversation["_reqid"]
                self.conversation_id = conversation["conversation_id"]
                self.response_id = conversation["response_id"]
                self.choice_id = conversation["choice_id"]
                self.SNlM0e = conversation["SNlM0e"]
                return True
        return False

    async def __get_snlm0e(self):
        # Find "SNlM0e":"<ID>"
        if not self.session_id or self.session_id[-1] != ".":
            raise Exception(
                "__Secure-1PSID value must end with a single dot. Enter correct __Secure-1PSID value.",
            )
        resp = await self.session.get(
            "https://bard.google.com/",
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Response code not 200. Response Status is {resp.status_code}",
            )
        SNlM0e = re.search(r"SNlM0e\":\"(.*?)\"", resp.text)
        if not SNlM0e:
            raise Exception(
                "SNlM0e value not found in response. Check __Secure-1PSID value.",
            )
        return SNlM0e.group(1)

    async def ask(self, message: str) -> dict:
        """
        Send a message to Google Bard and return the response.
        :param message: The message to send to Google Bard.
        :return: A dict containing the response from Google Bard.
        """
        # url params
        params = {
            "bl": "boq_assistant-bard-web-server_20230620.14_p0",
            "_reqid": str(self._reqid),
            "rt": "c",
        }

        # message arr -> data["f.req"]. Message is double json stringified
        message_struct = [
            [message],
            None,
            [self.conversation_id, self.response_id, self.choice_id],
        ]
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct)]),
            "at": self.SNlM0e,
        }
        resp = await self.session.post(
            "https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate",
            params=params,
            data=data,
            timeout=self.timeout,
        )
        chat_data = json.loads(resp.content.splitlines()[3])[0][2]
        if not chat_data:
            return {"content": f"Google Bard encountered an error: {resp.content}."}
        json_chat_data = json.loads(chat_data)
        images = []
        if len(json_chat_data) >= 3:
            if len(json_chat_data[4][0]) >= 4:
                if json_chat_data[4][0][4]:
                    for img in json_chat_data[4][0][4]:
                        images.append(img[0][0][0])
        results = {
            "content": json_chat_data[4][0][1][0],
            "conversation_id": json_chat_data[1][0],
            "response_id": json_chat_data[1][1],
            "factualityQueries": json_chat_data[3],
            "textQuery": json_chat_data[2][0] if json_chat_data[2] is not None else "",
            "choices": [{"id": i[0], "content": i[1]} for i in json_chat_data[4]],
            "images": images,
        }
        self.conversation_id = results["conversation_id"]
        self.response_id = results["response_id"]
        self.choice_id = results["choices"][0]["id"]
        self._reqid += 100000
        return results

@register_engine
@singleton
class BardEngine(LLMChat):
    name = ("Bard", "bard")

    def __init__(
        self,
    ) -> None:
        super().__init__(device="api")

        self._chatbot = Chatbot(session_id=os.environ.get("GOOGLE_BARD_SESSION_ID", None))

    def call(
        self, conversation: Conversation, n_completions: int = 1, **kwargs: Any
    ) -> List[ConversationTurn]:

        if len(conversation.turns) != 1:
            raise AssertionError("BARD conversations must have exactly one turn.")

        # Extract the last user turn from the conversation.
        user_turn = conversation.turns[-1]
        new_prompt = user_turn.text
        if user_turn.speaker in (Speaker.SYSTEM, Speaker.AI):
            raise AssertionError("The last turn in the conversation must be from the user.")

        outputs = []
        for _ in range(n_completions):
            logging.info("Sending prompt to BARD engine: %s", new_prompt)
            outputs.append(self._chatbot.ask(new_prompt)["content"])

        return [ConversationTurn(speaker=Speaker.AI, text=output) for output in outputs]
