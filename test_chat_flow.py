"""Quick test: create conversation, send message, verify persistence."""
import requests, json

BASE = "http://localhost:8000"

# 1. Create conversation
r = requests.post(f"{BASE}/api/conversations", json={"title": "New Chat"})
conv = r.json()
conv_id = conv["id"]
print(f"[1] Created conversation: {conv_id[:8]}...")

# 2. Send first message (non-streaming for simplicity)
r2 = requests.post(f"{BASE}/api/conversations/{conv_id}/chat",
                    json={"message": "Hi! I am Adith and I love building AI projects with Python", "top_k": 3},
                    stream=True)
full = ""
for line in r2.iter_lines():
    if not line:
        continue
    text = line.decode()
    if text.startswith("data: "):
        data = json.loads(text[6:])
        if data.get("type") == "token":
            full += data["content"]
        if data.get("type") == "title":
            print(f"[2] Auto-titled: {data['title']}")
        if data.get("type") == "metadata":
            print(f"[2] Retrieved {len(data['memories_used'])} memories in {data['retrieval_time_ms']}ms")
print(f"[2] AI response: {full[:150]}...")

# 3. Send second message
r3 = requests.post(f"{BASE}/api/conversations/{conv_id}/chat",
                    json={"message": "What is my name and what do I like?", "top_k": 5},
                    stream=True)
full2 = ""
for line in r3.iter_lines():
    if not line:
        continue
    text = line.decode()
    if text.startswith("data: "):
        data = json.loads(text[6:])
        if data.get("type") == "token":
            full2 += data["content"]
        if data.get("type") == "metadata":
            mems = data["memories_used"]
            print(f"\n[3] Retrieved {len(mems)} memories:")
            for m in mems[:3]:
                print(f"    [{m['type']}] score={m['score']}: {m['content'][:80]}")
print(f"\n[3] AI response: {full2[:200]}...")

# 4. Verify persistence - load conversation from DB
r4 = requests.get(f"{BASE}/api/conversations/{conv_id}")
conv_data = r4.json()
print(f"\n[4] Conversation title: {conv_data['title']}")
print(f"[4] Total messages stored: {len(conv_data['messages'])}")
for msg in conv_data["messages"]:
    print(f"    [{msg['role']}] {msg['content'][:80]}...")

# 5. List all conversations
r5 = requests.get(f"{BASE}/api/conversations")
convs = r5.json()
print(f"\n[5] Total conversations: {len(convs)}")
for c in convs:
    print(f"    {c['title']} ({c['message_count']} messages)")
