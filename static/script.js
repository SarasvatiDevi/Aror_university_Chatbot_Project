class ArorUniversityChatbot {
    constructor() {
        this.chatMessages = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-input');
        this.sendButton = document.getElementById('send-button');

        this.apiBaseUrl = window.location.origin;
        this.apiEndpoint = '/api/ask';
        this.jsonData = null;

        this.init();
    }

    async init() {
        await this.loadJSONData();
        this.setupEventListeners();
        this.userInput.focus();
    }

    async loadJSONData() {
        try {
            // Assuming your JSON is available as a variable or in a file
            // If it's in a file, use: const response = await fetch('./aror_data.json');
            // For now, I'll assume the JSON is assigned to a variable called `arorData`

            if (typeof arorData !== 'undefined') {
                this.jsonData = arorData;
                console.log('JSON data loaded successfully:', this.jsonData.length + ' questions loaded');
            } else {
                // Fallback: use the JSON you provided directly
                this.jsonData = [
                  {
                    "id": "1",
                    "question": "What is the name of the university?",
                    "answer": "Aror University of Art, Architecture, Design & Heritage, Sukkur"
                  },
                  // ... include all your JSON data here
                  // For production, better to load from external file
                ];
                console.log('Using embedded JSON data:', this.jsonData.length + ' questions loaded');
            }
        } catch (error) {
            console.error('Error loading JSON data:', error);
            this.jsonData = [];
        }
    }

    setupEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key in input
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Input focus for better UX
        this.userInput.addEventListener('focus', () => {
            this.userInput.parentElement.classList.add('focused');
        });

        this.userInput.addEventListener('blur', () => {
            this.userInput.parentElement.classList.remove('focused');
        });

        // Auto-resize input
        this.userInput.addEventListener('input', this.autoResizeInput.bind(this));
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            let response;
            // Try backend API first
            try {
                response = await this.getBotResponse(message);
            } catch (apiError) {
                console.log('API failed, using JSON fallback:', apiError);
                response = this.getResponseFromJSON(message);
            }

            this.removeTypingIndicator();
            this.addMessage(response, 'bot');
        } catch (error) {
            console.error('Error:', error);
            this.removeTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again or contact the university directly at 📱 0325-2278377.', 'bot');
        }
    }

    async getBotResponse(message) {
        const response = await fetch(this.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message
            })
        });

        if (!response.ok) {
            throw new Error(`API response error: ${response.status}`);
        }

        const data = await response.json();
        return data.answer;
    }

    getResponseFromJSON(message) {
        if (!this.jsonData || this.jsonData.length === 0) {
            return this.getDefaultFallbackResponse(message);
        }

        const lowerMessage = message.toLowerCase().trim();

        // First, try exact match
        const exactMatch = this.jsonData.find(item =>
            item.question.toLowerCase().trim() === lowerMessage
        );

        if (exactMatch) {
            return exactMatch.answer;
        }

        // Try partial matches with scoring
        const matches = this.jsonData.map(item => {
            const itemQuestion = item.question.toLowerCase();
            let score = 0;

            // Check if user message contains the question or vice versa
            if (lowerMessage.includes(itemQuestion) || itemQuestion.includes(lowerMessage)) {
                score += 10;
            }

            // Check word overlap
            const userWords = new Set(lowerMessage.split(/\s+/));
            const itemWords = new Set(itemQuestion.split(/\s+/));
            const commonWords = [...userWords].filter(word =>
                itemWords.has(word) && word.length > 2 // ignore short words
            );

            score += commonWords.length * 2;

            // Bonus for matches at the beginning
            if (itemQuestion.startsWith(lowerMessage) || lowerMessage.startsWith(itemQuestion)) {
                score += 5;
            }

            return { item, score };
        }).filter(match => match.score > 0)
          .sort((a, b) => b.score - a.score);

        if (matches.length > 0 && matches[0].score >= 3) {
            return matches[0].item.answer;
        }

        // Check for greetings and common phrases
        const greetingResponse = this.checkForGreetings(lowerMessage);
        if (greetingResponse) {
            return greetingResponse;
        }

        return this.getDefaultFallbackResponse(message);
    }

    checkForGreetings(message) {
        const greetings = {
            'hello': 'Hello! Welcome to Aror University. How can I assist you today?',
            'hi': 'Hi there! I\'m here to help you learn about Aror University. What would you like to know?',
            'hey': 'Hey! How can I help you with Aror University today?',
            'good morning': 'Good morning! Welcome to Aror University assistant. How can I help you?',
            'good afternoon': 'Good afternoon! How can I assist you with Aror University?',
            'good evening': 'Good evening! What would you like to know about Aror University?',
            'thanks': 'You\'re welcome! Feel free to ask if you have any other questions about Aror University.',
            'thank you': 'You\'re welcome! If you need more information, don\'t hesitate to ask.',
            'thank': 'You\'re welcome! Is there anything else I can help you with?',
            'how are you': 'I\'m doing great, thanks for asking! How can I help you with Aror University today?',
            'bye': 'Goodbye! Have a great day. Feel free to come back if you have more questions about Aror University!',
            'goodbye': 'Goodbye! Have a great day. Feel free to come back if you have more questions about Aror University!'
        };

        for (const [greeting, response] of Object.entries(greetings)) {
            if (message.includes(greeting)) {
                return response;
            }
        }

        return null;
    }

    getDefaultFallbackResponse(message) {
        return `I'm not sure I understand your question about Aror University. Could you please rephrase it?

You can ask about:
• Admissions and requirements
• Courses and departments
• Fee structure
• Faculty information
• Location and contact details
• Hostel facilities
• Scholarships

For specific queries, please contact us directly at 📱 0325-2278377 or email admissions@aror.edu.pk`;
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-sender">${sender === 'user' ? 'You' : 'Aror Assistant'}</div>
                ${this.formatMessage(content)}
            </div>
            <div class="message-time">${timestamp}</div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessage(content) {
        // Convert URLs to clickable links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        let formattedContent = content.replace(urlRegex, url =>
            `<a href="${url}" target="_blank" class="message-link">${url}</a>`
        );

        // Convert line breaks to HTML
        formattedContent = formattedContent.replace(/\n/g, '<br>');

        return formattedContent;
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    autoResizeInput() {
        this.userInput.style.height = 'auto';
        this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ArorUniversityChatbot();
});
