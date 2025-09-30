class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector('.chatbox__button button'),
      chatBox: document.querySelector('.chatbox__support'),
      sendButton: document.querySelector('.send__button'),
      input: document.querySelector('.chatbox__input'),
      quickReplies: document.querySelectorAll('.quick-btn')
    };
    this.state = false;
    this.messages = [];
    this._init();
  }

  _init() {
    const { openButton, chatBox, sendButton, input, quickReplies } = this.args;

    openButton.addEventListener('click', () => this.toggleState(chatBox));
    sendButton.addEventListener('click', () => this.onSendButton());
    input.addEventListener('keyup', (e) => { if (e.key === 'Enter') this.onSendButton(); });
    quickReplies.forEach(btn => btn.addEventListener('click', () => {
      input.value = btn.dataset.value;
      this.onSendButton();
    }));
  }

  toggleState(chatbox) {
    this.state = !this.state;
    chatbox.classList.toggle('chat-open');
    if (this.state && !this._welcomed) {
      this._welcomed = true;
      this._pushBotMessage("Xin chào! Mình là trợ lý y tế, hôm nay mình có thể giúp gì cho bạn?");
    }
  }

  _pushUserMessage(text) {
    this.messages.push({ name: 'User', message: text });
  }
  _pushBotMessage(text) {
    this.messages.push({ name: 'Bot', message: text });
    this.updateChatText();
  }
  _pushBotTyping() {
    this.messages.push({ name: 'Bot', typing: true });
    this.updateChatText();
  }
  _replaceBotTypingWithMessage(text) {
    const idx = this.messages.findIndex(m => m.typing);
    if (idx !== -1) this.messages.splice(idx, 1);
    this.messages.push({ name: 'Bot', message: text });
    this.updateChatText();
  }

  onSendButton() {
    const input = this.args.input;
    const text = input.value.trim();
    if (!text) return;
    this._pushUserMessage(text);
    input.value = '';
    this.updateChatText();
    this._pushBotTyping();

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    })
    .then(r => r.json())
    .then(r => {
      const answer = r.answer || "Xin lỗi, hiện không có câu trả lời.";
      this._replaceBotTypingWithMessage(answer);
    })
    .catch(err => {
      console.error(err);
      this._replaceBotTypingWithMessage("⚠️ Lỗi kết nối server.");
    });
  }

  updateChatText() {
    const chatbox = this.args.chatBox;
    const container = chatbox.querySelector('.chatbox__messages');
    const botAvatar = chatbox.dataset.botAvatar;
    const userAvatar = chatbox.dataset.userAvatar;

    container.innerHTML = this.messages.map(m => {
      const isBot = m.name === 'Bot';
      const rowClass = 'message-row ' + (isBot ? 'bot' : 'user');
      const bubbleClass = 'messages__item ' + (isBot ? 'messages__item--bot' : 'messages__item--user');
      const avatar = `<div class="avatar"><img src="${isBot ? botAvatar : userAvatar}" alt="${m.name}"></div>`;
      const content = m.typing ? `<span class="typing"><span></span><span></span><span></span></span>` : m.message;
      return `<div class="${rowClass}">${isBot ? avatar : ''}<div class="${bubbleClass}">${content}</div>${!isBot ? avatar : ''}</div>`;
    }).join('');
    container.scrollTop = container.scrollHeight;
  }
}

const chatbox = new Chatbox();
