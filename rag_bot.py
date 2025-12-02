import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import os
import time


class ProductionRAGBot:
    def __init__(self, json_file_path: str = None):
        """
        Production-ready RAG system for Aror University with spelling correction
        """
        self.qa_pairs = []
        self.ai_enabled = True
        self.initialization_time = time.time()

        # Spelling corrections dictionary
        self.spelling_corrections = {
            'avialable': 'available',
            'semseter': 'semester',
            'univetsity': 'university',
            'archaelogy': 'archaeology',
            'archeology': 'archaeology',
            'schema': 'scheme',
            'desing': 'design',
            'enginnering': 'engineering',
            'artifical': 'artificial',
            'inteligence': 'intelligence',
            'multimedya': 'multimedia',
            'facalty': 'faculty',
            'departmant': 'department',
            'chairmen': 'chairman',
            'in charge': 'incharge',
            'admision': 'admission',
            'eligiblity': 'eligibility',
            'criterias': 'criteria',
            'hostory': 'history',
            'textile': 'textile',
            'fashion': 'fashion',
            'civil': 'civil',
            'environmental': 'environmental',
            'aror': 'aror',
            'gamming': 'gaming',
            'multimedya': 'multimedia',
            'cyber': 'cyber',
            'sceince': 'science',
            'sciences': 'science',
            'pakistan': 'pakistan',
            'studies': 'studies',
            'tourism': 'tourism',
            'hospitality': 'hospitality',
            'visual': 'visual',
            'arts': 'arts',
            'digital': 'digital',
            'ceramic': 'ceramic',
            'fee': 'fee',
            'fees': 'fee',
            'structure': 'structure',
            'sample': 'sample',
            'paper': 'paper',
            'entry': 'entry',
            'test': 'test',
            'vc': 'vc',
            'hod': 'hod',
            'incharge': 'incharge',
            'office': 'office',
            'location': 'location',
            'where': 'where',
            'faculty': 'faculty',
            'duration': 'duration',
            'program': 'program',
            'bs': 'bs',
            'eligibility': 'eligibility',
            'criteria': 'criteria',
            'gpa': 'gpa',
            'cgpa': 'cgpa',
            'marks': 'marks',
            'pass': 'pass',
            'subject': 'subject',
            'hostel': 'hostel',
            'boys': 'boys',
            'girls': 'girls',
            'facilities': 'facilities'
        }

        # Use default path if none provided
        if json_file_path is None:
            json_file_path = self.find_json_file()

        print(f" Loading data from: {json_file_path}")
        self.data = self.load_data(json_file_path)

        if not self.data:
            print(" No data loaded. Using fallback mode.")
            self.ai_enabled = False
            return

        self.questions = [item['question'].lower().strip() for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        self.qa_pairs = self.data

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )

        # Fit the vectorizer and transform questions
        try:
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
            print(f" Vectorizer trained with {len(self.questions)} questions")
        except Exception as e:
            print(f"Error initializing vectorizer: {e}")
            self.ai_enabled = False
            return

        # Create question-answer mapping for exact matches
        self.qa_mapping = {
            self.normalize_text(item['question']): item['answer']
            for item in self.data
        }

        # Enhanced synonyms
        self.synonyms = {
            'vc': ['vice chancellor', 'vc','authority'],
            'hod': ['head of department', 'chairman', 'hod'],
            'ai': ['artificial intelligence', 'ai','artificial-intelligence'],
            'mmg': ['multimedia gaming', 'multimedia and gaming', 'mmg','multi media gaming'],
            'bs': ['bachelor', 'bs', 'bachelor of science'],
            'hostel': ['hostel', 'dormitory', 'accommodation'],
            'fee': ['fee', 'fees', 'cost', 'charges'],
            'admission': ['admission', 'admissions', 'apply'],
            'scheme': ['scheme', 'syllabus', 'curriculum', 'schema', 'course outline'],
            'timing': ['timing', 'timings', 'hours', 'schedule','time'],
            'location': ['location', 'address', 'where', 'situated'],
            'established': ['established', 'founded', 'started','establish'],
            'recognized': ['recognized', 'recognition', 'accredited'],
            'department': ['department', 'faculty', 'program', 'course','dept'],
            'use':['used','using','follow','followed'],
            'stay':['secure','secured'],
            'mathematics': ['mathematics', 'math', 'mathematic', 'mathematics'],
            'papers': ['papers', 'paper', 'paper', 'papers', 'pepers', 'pepar'],
            'uni':['university','uni','univarsity'],
            'where':['wheres',"where's",'where'],
            'what':['whats', "what's",'what'],
            'cybersecurity':['cyber security','cyber-security','cybersecurity'],
            'cyber security': ['cyber security', 'cyber-security', 'cybersecurity']

        }

        self.initialization_time = time.time() - self.initialization_time
        print(f"Production RAG Bot with Spelling Correction ready! Initialized in {self.initialization_time:.2f}s")

    def correct_spelling(self, text: str) -> str:
        """Correct common spelling mistakes in the query"""
        words = text.split()
        corrected_words = []

        for word in words:
            # Check if word needs correction
            if word in self.spelling_corrections:
                corrected_words.append(self.spelling_corrections[word])
                print(f"Spelling corrected: '{word}' -> '{self.spelling_corrections[word]}'")
            else:
                corrected_words.append(word)

        corrected_text = ' '.join(corrected_words)

        # If the entire text was changed, log it
        if corrected_text != text:
            print(f"Query spelling corrected: '{text}' -> '{corrected_text}'")

        return corrected_text

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching with spelling correction"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Apply spelling correction
        text = self.correct_spelling(text)

        return text

    def find_json_file(self):
        """Find the JSON file in common locations"""
        possible_paths = [
            'university_data.json',
            'data/university_data.json',
            '../university_data.json',
            './data/university_data.json',
            os.path.join(os.path.dirname(__file__), 'university_data.json'),
            os.path.join(os.path.dirname(__file__), 'data', 'university_data.json')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found data file at: {path}")
                return path

        print("No JSON file found in common locations")
        return None

    def load_data(self, file_path: str) -> List[Dict]:
        """Load and clean JSON data"""
        try:
            if not file_path or not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return []

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Clean the data
            cleaned_data = []
            for item in data:
                if 'question' in item and 'answer' in item:
                    cleaned_item = {
                        'id': item.get('id', ''),
                        'question': str(item['question']).strip(),
                        'answer': str(item['answer']).strip()
                    }
                    cleaned_data.append(cleaned_item)

            print(f"Loaded {len(cleaned_data)} Q&A pairs")
            return cleaned_data

        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms"""
        expanded_query = query.lower()
        for key, synonyms in self.synonyms.items():
            for synonym in synonyms:
                if synonym in expanded_query and key not in expanded_query:
                    expanded_query += " " + key
        return expanded_query

    def find_exact_match(self, query: str) -> Optional[str]:
        """Find exact match for query"""
        if not self.ai_enabled:
            return None

        normalized_query = self.normalize_text(query)

        # Direct match
        if normalized_query in self.qa_mapping:
            return self.qa_mapping[normalized_query]

        # Check for close matches
        for question in self.qa_mapping.keys():
            if self.calculate_similarity(normalized_query, question) > 0.95:
                return self.qa_mapping[question]

        return None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between texts"""
        try:
            vector1 = self.vectorizer.transform([text1])
            vector2 = self.vectorizer.transform([text2])
            return cosine_similarity(vector1, vector2)[0][0]
        except:
            return 0.0

    def find_semantic_match(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find semantic matches using TF-IDF"""
        if not self.ai_enabled:
            return []

        expanded_query = self.expand_query_with_synonyms(query)

        try:
            query_vector = self.vectorizer.transform([expanded_query])
            similarities = cosine_similarity(query_vector, self.question_vectors)[0]

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []

            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append((self.answers[idx], similarities[idx]))

            return results
        except:
            return []

    def get_response(self, query: str) -> Dict:
        """
        Main method to get response - compatible with FastAPI
        Returns: {
            'answer': str,
            'confidence': float,
            'type': str,
            'sources': list,
            'timestamp': str
        }
        """
        if not self.ai_enabled:
            return {
                'answer': "🤖 System is in fallback mode. Please contact us directly: 📱 0325-2278377 | 📧 admissions@aror.edu.pk",
                'confidence': 0.0,
                'type': 'fallback',
                'sources': [],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Handle empty query
        if not query or not query.strip():
            return {
                'answer': "Please ask a question about Aror University!",
                'confidence': 0.0,
                'type': 'fallback',
                'sources': [],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Apply spelling correction to the original query
        original_query = query.strip()
        query = self.correct_spelling(original_query)

        # Casual responses
        casual_responses = {
            'hi': "Hi there! Ask me anything about Aror University. 🎓",
            'hello': "Hello! How can I help you with Aror University today?",
            'hwllo': "Hello! How can I help you with Aror University today?",
            'hllo': "Hello! How can I help you with Aror University today?",
            'how are you': "I'm doing great! Ready to help you with Aror University information.",
            'thanks': "You're welcome! Happy to help. 😊",
            'thank you': "You're welcome! Feel free to ask more questions.",
            'bye': "Goodbye! Have a great day! 👋",
            'goodbye': "Goodbye! Visit again for more information about Aror University."
        }

        normalized_query = query.lower().strip()
        if normalized_query in casual_responses:
            return {
                'answer': casual_responses[normalized_query],
                'confidence': 1.0,
                'type': 'casual',
                'sources': [],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Try exact match
        exact_answer = self.find_exact_match(query)
        if exact_answer:
            return {
                'answer': exact_answer,
                'confidence': 1.0,
                'type': 'exact',
                'sources': [{'type': 'exact_match', 'score': 1.0}],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Try semantic matching
        semantic_matches = self.find_semantic_match(query)
        if semantic_matches:
            best_answer, confidence = semantic_matches[0]
            if confidence >= 0.3:
                sources = [{'type': 'semantic_match', 'score': float(conf)} for _, conf in semantic_matches[:3]]
                return {
                    'answer': best_answer,
                    'confidence': float(confidence),
                    'type': 'semantic',
                    'sources': sources,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }

        # Fallback response
        return {
            'answer': "I'm still learning about that topic. For specific questions, please contact Aror University directly:\n\n📞 Phone: 0325-2278377\n📧 Email: admissions@aror.edu.pk\n🌐 Website: https://aror.edu.pk/",
            'confidence': 0.0,
            'type': 'fallback',
            'sources': [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_system_status(self) -> Dict:
        """Get system status for monitoring"""
        return {
            'ai_enabled': self.ai_enabled,
            'qa_pairs_count': len(self.qa_pairs),
            'initialization_time': self.initialization_time,
            'vectorizer_ready': hasattr(self, 'question_vectors'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

#
# import json
# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict, Tuple, Optional
# import os
# import time
#
#
# class ProductionRAGBot:
#     def __init__(self, json_file_path: str = None):
#         """
#         Production-ready RAG system with improved variations
#         """
#         self.qa_pairs = []
#         self.ai_enabled = True
#         self.initialization_time = time.time()
#
#         # Enhanced spelling corrections
#         self.spelling_corrections = {
#             'avialable': 'available',
#             'semseter': 'semester',
#             'univetsity': 'university',
#             'archaelogy': 'archaeology',
#             'archeology': 'archaeology',
#             'schema': 'scheme',
#             'desing': 'design',
#             'enginnering': 'engineering',
#             'artifical': 'artificial',
#             'inteligence': 'intelligence',
#             'multimedya': 'multimedia',
#             'facalty': 'faculty',
#             'departmant': 'department',
#             'chairmen': 'chairman',
#             'in charge': 'incharge',
#             'admision': 'admission',
#             'eligiblity': 'eligibility',
#             'criterias': 'criteria',
#             'hostory': 'history',
#             'gamming': 'gaming',
#             'sceince': 'science',
#             'hwllo': 'hello',
#             'hllo': 'hello',
#             'cybersecurity': 'cyber security',
#             'cyber-security': 'cyber security'
#         }
#
#         # Use default path if none provided
#         if json_file_path is None:
#             json_file_path = self.find_json_file()
#
#         print(f" Loading data from: {json_file_path}")
#         self.data = self.load_data(json_file_path)
#
#         if not self.data:
#             print(" No data loaded. Using fallback mode.")
#             self.ai_enabled = False
#             return
#
#         # Store original questions and answers
#         self.questions = [item['question'].lower().strip() for item in self.data]
#         self.answers = [item['answer'] for item in self.data]
#         self.qa_pairs = self.data
#
#         # Create enhanced question-answer mapping with improved variations
#         self.qa_mapping = {}
#         self.build_smart_variations()
#
#         # Initialize TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer(
#             lowercase=True,
#             stop_words='english',
#             ngram_range=(1, 2),
#             min_df=1,
#             max_df=0.9
#         )
#
#         # Fit the vectorizer with all questions
#         all_questions = list(self.qa_mapping.keys())
#         try:
#             self.question_vectors = self.vectorizer.fit_transform(all_questions)
#             print(f" Vectorizer trained with {len(all_questions)} questions")
#         except Exception as e:
#             print(f"Error initializing vectorizer: {e}")
#             self.ai_enabled = False
#             return
#
#         self.initialization_time = time.time() - self.initialization_time
#         print(f"Production RAG Bot ready! Initialized in {self.initialization_time:.2f}s")
#
#     def build_smart_variations(self):
#         """Build smart variations with better conflict resolution"""
#         for item in self.data:
#             original_question = item['question']
#             answer = item['answer']
#
#             # Add original question
#             normalized_original = self.normalize_text(original_question)
#             self.qa_mapping[normalized_original] = answer
#
#             # Generate smart variations
#             variations = self.generate_smart_variations(original_question, answer)
#             for variation in variations:
#                 normalized_variation = self.normalize_text(variation)
#
#                 # Only add if not already present or if it's a better match
#                 if normalized_variation not in self.qa_mapping:
#                     self.qa_mapping[normalized_variation] = answer
#                 else:
#                     # If already exists, check if this is a better match (scheme vs leadership)
#                     existing_answer = self.qa_mapping[normalized_variation]
#                     if self.is_scheme_question(variation) and not self.is_scheme_answer(existing_answer):
#                         self.qa_mapping[normalized_variation] = answer
#
#         print(f"Built mapping with {len(self.qa_mapping)} variations")
#
#     def generate_smart_variations(self, question: str, answer: str) -> List[str]:
#         """Generate smart variations with context awareness"""
#         variations = []
#         question_lower = question.lower()
#
#         # Skip for simple questions
#         if any(word in question_lower for word in ['hi', 'hello', 'thanks', 'bye']):
#             return variations
#
#         # Detect question type from both question and answer
#         question_type = self.detect_question_type(question_lower, answer)
#         department = self.extract_department(question_lower)
#
#         # Generate variations based on detected type
#         if question_type == "leadership":
#             variations.extend(self.generate_leadership_variations(question_lower, department))
#         elif question_type == "scheme":
#             variations.extend(self.generate_scheme_variations(question_lower, department))
#         elif question_type == "eligibility":
#             variations.extend(self.generate_eligibility_variations(question_lower, department))
#
#         # Always generate structural variations
#         variations.extend(self.generate_structural_variations(question_lower))
#
#         return list(set([v for v in variations if v and v != question_lower]))
#
#     def generate_leadership_variations(self, question: str, department: str) -> List[str]:
#         """Generate variations for leadership questions"""
#         variations = []
#
#         if not department:
#             return variations
#
#         # Role variations
#         role_variations = {
#             'chairman': ['incharge', 'head'],
#             'incharge': ['chairman', 'head'],
#             'head': ['chairman', 'incharge']
#         }
#
#         for role, alts in role_variations.items():
#             if role in question:
#                 for alt_role in alts:
#                     variations.append(f"who is {alt_role} of {department}")
#                     variations.append(f"who is the {alt_role} of {department}")
#
#         # Department variations
#         dept_variations = {
#             "artificial intelligence": ["ai"],
#             "multimedia gaming": ["mmg"],
#             "cyber security": ["cybersecurity"],
#             "visual arts": ["visual art"]
#         }
#
#         if department in dept_variations:
#             for dept_var in dept_variations[department]:
#                 variations.append(f"who is chairman of {dept_var}")
#                 variations.append(f"who is incharge of {dept_var}")
#
#         return variations
#
#     def generate_scheme_variations(self, question: str, department: str) -> List[str]:
#         """Generate variations for scheme questions"""
#         variations = []
#
#         if not department:
#             return variations
#
#         scheme_terms = ['scheme', 'syllabus', 'curriculum', 'course outline']
#
#         for term in scheme_terms:
#             if term not in question:
#                 variations.append(f"{term} of {department}")
#                 variations.append(f"{department} {term}")
#
#         # URL detection for scheme answers
#         if any(url_indicator in question for url_indicator in ['scheme', 'syllabus']):
#             for dept_var in [department, 'cyber security', 'cybersecurity']:
#                 variations.append(f"schema of {dept_var}")
#                 variations.append(f"{dept_var} schema")
#
#         return variations
#
#     def generate_eligibility_variations(self, question: str, department: str) -> List[str]:
#         """Generate variations for eligibility questions"""
#         variations = []
#
#         if not department:
#             return variations
#
#         eligibility_terms = ['eligibility', 'criteria', 'requirement']
#
#         for term in eligibility_terms:
#             if term not in question:
#                 variations.append(f"{term} for {department}")
#                 variations.append(f"{department} {term}")
#
#         return variations
#
#     def generate_structural_variations(self, question: str) -> List[str]:
#         """Generate structural variations"""
#         variations = []
#
#         # "The" variations
#         if "who is " in question and "who is the " not in question:
#             variations.append(question.replace("who is ", "who is the "))
#         if "who is the " in question:
#             variations.append(question.replace("who is the ", "who is "))
#
#         # Department format variations
#         if "cyber security" in question:
#             variations.append(question.replace("cyber security", "cybersecurity"))
#         if "cybersecurity" in question:
#             variations.append(question.replace("cybersecurity", "cyber security"))
#
#         return variations
#
#     def detect_question_type(self, question: str, answer: str) -> str:
#         """Detect question type from both question and answer"""
#         # Check answer content first
#         if any(url_indicator in answer for url_indicator in ['.pdf', '.edu.pk', 'http']):
#             return "scheme"
#         elif any(role in answer for role in ['prof', 'dr', 'mr', 'ms']):
#             return "leadership"
#         elif any(eligibility_indicator in answer for eligibility_indicator in ['marks', 'percentage', 'criteria']):
#             return "eligibility"
#
#         # Fallback to question content
#         if any(word in question for word in ['scheme', 'syllabus', 'curriculum']):
#             return "scheme"
#         elif any(word in question for word in ['chairman', 'incharge', 'dean', 'head']):
#             return "leadership"
#         elif any(word in question for word in ['eligibility', 'criteria', 'requirement']):
#             return "eligibility"
#         else:
#             return "general"
#
#     def is_scheme_question(self, question: str) -> bool:
#         """Check if question is about scheme/syllabus"""
#         return any(word in question for word in ['scheme', 'syllabus', 'curriculum', 'schema'])
#
#     def is_scheme_answer(self, answer: str) -> bool:
#         """Check if answer is a scheme/syllabus URL"""
#         return any(url_indicator in answer for url_indicator in ['.pdf', '.edu.pk', 'http'])
#
#     def extract_department(self, question: str) -> str:
#         """Extract department name from question"""
#         departments = [
#             "fashion design", "artificial intelligence", "multimedia gaming",
#             "civil engineering", "visual arts", "textile design", "cyber security",
#             "environmental science", "history", "architecture", "archaeology",
#             "pakistan studies", "tourism hospitality"
#         ]
#
#         for dept in departments:
#             if dept in question:
#                 return dept
#         return ""
#
#     def correct_spelling(self, text: str) -> str:
#         """Correct common spelling mistakes"""
#         words = text.split()
#         corrected_words = []
#
#         for word in words:
#             if word in self.spelling_corrections:
#                 corrected_words.append(self.spelling_corrections[word])
#             else:
#                 corrected_words.append(word)
#
#         return ' '.join(corrected_words)
#
#     def normalize_text(self, text: str) -> str:
#         """Normalize text for matching"""
#         text = text.lower().strip()
#         text = re.sub(r'[^\w\s]', ' ', text)
#         text = re.sub(r'\s+', ' ', text)
#         text = self.correct_spelling(text)
#         return text
#
#     def find_json_file(self):
#         """Find the JSON file"""
#         possible_paths = [
#             'university_data.json',
#             'data/university_data.json',
#             '../university_data.json',
#             './data/university_data.json',
#             os.path.join(os.path.dirname(__file__), 'university_data.json'),
#             os.path.join(os.path.dirname(__file__), 'data', 'university_data.json')
#         ]
#
#         for path in possible_paths:
#             if os.path.exists(path):
#                 print(f"Found data file at: {path}")
#                 return path
#
#         print("No JSON file found")
#         return None
#
#     def load_data(self, file_path: str) -> List[Dict]:
#         """Load and clean JSON data"""
#         try:
#             if not file_path or not os.path.exists(file_path):
#                 return []
#
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#
#             cleaned_data = []
#             for item in data:
#                 if 'question' in item and 'answer' in item:
#                     cleaned_item = {
#                         'id': item.get('id', ''),
#                         'question': str(item['question']).strip(),
#                         'answer': str(item['answer']).strip()
#                     }
#                     cleaned_data.append(cleaned_item)
#
#             print(f"Loaded {len(cleaned_data)} Q&A pairs")
#             return cleaned_data
#
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return []
#
#     def find_exact_match(self, query: str) -> Optional[str]:
#         """Find exact match for query"""
#         if not self.ai_enabled:
#             return None
#
#         normalized_query = self.normalize_text(query)
#
#         # Direct match
#         if normalized_query in self.qa_mapping:
#             return self.qa_mapping[normalized_query]
#
#         return None
#
#     def calculate_similarity(self, text1: str, text2: str) -> float:
#         """Calculate similarity between texts"""
#         try:
#             vector1 = self.vectorizer.transform([text1])
#             vector2 = self.vectorizer.transform([text2])
#             similarity = cosine_similarity(vector1, vector2)[0][0]
#             return similarity
#         except:
#             return 0.0
#
#     def find_semantic_match(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
#         """Find semantic matches using TF-IDF"""
#         if not self.ai_enabled:
#             return []
#
#         normalized_query = self.normalize_text(query)
#
#         try:
#             query_vector = self.vectorizer.transform([normalized_query])
#             similarities = cosine_similarity(query_vector, self.question_vectors)[0]
#
#             top_indices = np.argsort(similarities)[-top_k:][::-1]
#             results = []
#
#             for idx in top_indices:
#                 if similarities[idx] > 0.5:  # Higher threshold for better accuracy
#                     question = list(self.qa_mapping.keys())[idx]
#                     results.append((self.qa_mapping[question], similarities[idx]))
#
#             return results
#         except Exception as e:
#             print(f"Error in semantic matching: {e}")
#             return []
#
#     def get_response(self, query: str) -> Dict:
#         """
#         Main method to get response
#         """
#         if not self.ai_enabled:
#             return {
#                 'answer': "🤖 System is in fallback mode. Please contact us directly: 📱 0325-2278377",
#                 'confidence': 0.0,
#                 'type': 'fallback',
#                 'sources': [],
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#             }
#
#         if not query or not query.strip():
#             return {
#                 'answer': "Please ask a question about Aror University!",
#                 'confidence': 0.0,
#                 'type': 'fallback',
#                 'sources': [],
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#             }
#
#         # Apply spelling correction
#         original_query = query.strip()
#         query = self.correct_spelling(original_query)
#
#         # Casual responses
#         casual_responses = {
#             'hi': "Hi there! Ask me anything about Aror University. 🎓",
#             'hello': "Hello! How can I help you with Aror University today?",
#             'hwllo': "Hello! How can I help you with Aror University today?",
#             'hllo': "Hello! How can I help you with Aror University today?",
#             'how are you': "I'm doing great! Ready to help you with Aror University information.",
#             'thanks': "You're welcome! Happy to help. 😊",
#             'thank you': "You're welcome! Feel free to ask more questions.",
#             'bye': "Goodbye! Have a great day! 👋",
#             'goodbye': "Goodbye! Visit again for more information about Aror University."
#         }
#
#         normalized_query = query.lower().strip()
#         if normalized_query in casual_responses:
#             return {
#                 'answer': casual_responses[normalized_query],
#                 'confidence': 1.0,
#                 'type': 'casual',
#                 'sources': [],
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#             }
#
#         # Try exact match first
#         exact_answer = self.find_exact_match(query)
#         if exact_answer:
#             return {
#                 'answer': exact_answer,
#                 'confidence': 1.0,
#                 'type': 'exact',
#                 'sources': [{'type': 'exact_match', 'score': 1.0}],
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#             }
#
#         # Try semantic matching
#         semantic_matches = self.find_semantic_match(query)
#         if semantic_matches:
#             best_answer, confidence = semantic_matches[0]
#             return {
#                 'answer': best_answer,
#                 'confidence': float(confidence),
#                 'type': 'semantic',
#                 'sources': [{'type': 'semantic_match', 'score': float(conf)} for _, conf in semantic_matches[:3]],
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#             }
#
#         # Final fallback
#         return {
#             'answer': "I'm still learning about that topic. For specific questions, please contact Aror University directly:\n\n📞 Phone: 0325-2278377\n📧 Email: admissions@aror.edu.pk\n🌐 Website: https://aror.edu.pk/",
#             'confidence': 0.0,
#             'type': 'fallback',
#             'sources': [],
#             'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
#         }
#
#
# # Test the bot
# if __name__ == "__main__":
#     bot = ProductionRAGBot()
#
#     test_queries = [
#         "hwllo",
#         "hello",
#         "who is the incharge of cyber security",
#         "who is the chairman of cybersecurity",
#         "schema of cyber security",
#         "syllabus of cybersecurity"
#     ]
#
#     print("\n🧪 TESTING IMPROVED BOT")
#     print("=" * 60)
#
#     for query in test_queries:
#         response = bot.get_response(query)
#         print(f"Q: {query}")
#         print(f"A: {response['answer']}")
#         print(f"Type: {response['type']} | Confidence: {response['confidence']:.2f}")
#         print("-" * 60)