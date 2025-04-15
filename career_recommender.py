import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class CareerRecommendationSystem:
    def __init__(self, dataset_path='expanded_career_interests_original_dataset.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.career_categories = None
        self.career_descriptions = self.get_career_descriptions()
        self.career_resources = self.get_career_resources()
        self.load_dataset()
    
    def get_career_descriptions(self):
        """Dictionary of career descriptions"""
        return {
            # Technology careers
            'Software Engineer': "Designs, develops, and maintains software systems and applications. Works with various programming languages and frameworks to create efficient, scalable software solutions.",
            'Data Scientist': "Analyzes and interprets complex data to help organizations make better decisions. Combines statistics, mathematics, and programming to extract insights from data.",
            'Cybersecurity Analyst': "Protects computer systems and networks from information disclosure, theft, and damage. Monitors for security breaches and implements security measures.",
            'User Experience Designer': "Creates meaningful and relevant experiences for users. Designs the entire process of acquiring and integrating a product, including aspects of branding, design, usability, and function.",
            'Cloud Solutions Architect': "Designs and implements cloud computing solutions for organizations. Develops migration strategies, application designs, and manages cloud infrastructure.",
            'Machine Learning Engineer': "Develops artificial intelligence systems that can learn and apply knowledge without specific directions. Creates and implements machine learning algorithms.",
            'DevOps Engineer': "Combines software development and IT operations. Works to shorten the development lifecycle while delivering features, fixes, and updates frequently.",
            'Blockchain Developer': "Develops and implements blockchain architecture and solutions. Creates decentralized applications and smart contracts.",
            'AI Research Scientist': "Conducts research to advance the field of artificial intelligence. Develops new algorithms, approaches, and applications for AI systems.",
            'Network Security Specialist': "Focuses on protecting an organization's network infrastructure. Designs and implements security measures for network systems.",
            
            # Science careers
            'Biologist': "Studies living organisms and their relationship to the environment. May specialize in particular types of organisms or specific aspects of life.",
            'Physicist': "Studies matter, energy, and their interactions. Develops theories and models to explain the properties of the natural world.",
            'Environmental Scientist': "Studies the effects of human activity on the environment. Works to identify, control, or eliminate sources of pollutants or hazards affecting the environment.",
            'Geneticist': "Studies genes, genetic variations, and heredity. Works to understand how traits are passed from generation to generation.",
            'Chemical Engineer': "Applies principles of chemistry, physics, and mathematics to solve problems involving the production or use of chemicals and other products.",
            'Astronomer': "Studies celestial objects, space, and the physical universe. Observes, researches, and interprets astronomical phenomena.",
            'Forensic Scientist': "Applies scientific principles and techniques to the investigation of crimes. Collects and analyzes physical evidence.",
            'Marine Biologist': "Studies marine organisms and their behaviors and interactions with the environment. May specialize in certain species, behaviors, or ecosystems.",
            'Geologist': "Studies the physical structure and processes of the Earth. Examines rocks, minerals, and the processes that shape the Earth's surface.",
            'Zoologist': "Studies animals and their interactions with ecosystems. Analyzes animal behavior, genetics, and life processes.",
            
            # Healthcare careers
            'Doctor/Physician': "Diagnoses and treats injuries and illnesses. May specialize in specific areas of medicine such as cardiology, neurology, or pediatrics.",
            'Psychologist': "Studies cognitive, emotional, and social processes and behavior. Applies research to help improve processes and behaviors.",
            'Nurse Practitioner': "Provides advanced nursing care. Can prescribe medication, examine patients, diagnose illnesses, and provide treatments.",
            'Pharmacist': "Prepares and dispenses medications. Advises patients and healthcare professionals on the selection, dosages, and side effects of medications.",
            'Dentist': "Diagnoses and treats problems with teeth and tissues in the mouth. Works to prevent problems and improve patients' appearance and confidence.",
            'Physical Therapist': "Helps injured or ill people improve movement and manage pain. Develops plans using treatment techniques to promote ability to move, reduce pain, and restore function.",
            'Veterinarian': "Diagnoses and treats diseases and injuries in animals. May specialize in certain types of animals or in specific areas of medicine.",
            'Medical Researcher': "Conducts research aimed at improving overall human health. May work to develop new treatments, medications, or medical devices.",
            'Nutritionist': "Advises people on what to eat to lead a healthy lifestyle or achieve a specific health-related goal. Studies and communicates the effects of food and nutrition on health.",
            'Psychiatrist': "Specializes in mental health, including substance use disorders. Diagnoses and treats mental, emotional, and behavioral disorders.",
            
            # Creative careers
            'Graphic Designer': "Creates visual concepts, using computer software or by hand, to communicate ideas that inspire, inform, and captivate consumers.",
            'Writer/Author': "Develops written content for various media. May work on books, articles, scripts, or web content.",
            'Multimedia Artist': "Creates special effects, animation, or other visual images using film, video, computers, or other electronic tools and media.",
            'Film Director': "Directs the making of a film. Controls a film's artistic and dramatic aspects and visualizes the script while guiding the technical crew and actors.",
            'Video Game Designer': "Designs core gameplay elements including storylines, role-play mechanics, and character biographies. Balances the gameplay mechanics to ensure the intended experience.",
            'Interior Designer': "Plans, designs, and furnishes interiors of residential, commercial, or industrial buildings. Creates functional, safe, and aesthetically pleasing spaces.",
            'Fashion Designer': "Designs clothing, footwear, and accessories. Creates original garments, works with design teams, and oversees the creation of prototypes.",
            'Music Producer': "Oversees and manages the recording of an artist's music. Controls the recording sessions and guides the artists and technical team during the recording process.",
            'Animator': "Creates multiple images (frames) that create an illusion of movement when displayed in rapid sequence. Works in films, video games, television, or internet.",
            'Art Director': "Determines the overall visual appearance and how it communicates visually with its audience. Directs others who develop artwork or layouts.",
            
            # More careers (continuing for other categories)...
            'Teacher': "Educates students of various ages in different subjects. Plans lessons, assesses student progress, and creates an engaging learning environment.",
            'Financial Analyst': "Evaluates investment opportunities. Researches and analyzes financial information to help companies make business decisions.",
            'Digital Marketer': "Promotes brands using digital channels. Develops, implements, and manages marketing campaigns that promote a company and its products/services.",
            'Entrepreneur': "Starts and runs own business ventures. Identifies opportunities, secures resources, and builds organizations to capitalize on those opportunities.",
            
            # Adding more examples for different categories...
            'Journalist': "Researches, writes, and reports news stories. Investigates and presents information to help people understand events, issues, and trends.",
            'Civil Engineer': "Designs, builds, and maintains infrastructure projects and systems. Works on roads, buildings, airports, tunnels, dams, bridges, and water supply systems.",
            'Sociologist': "Studies society and social behavior. Examines groups, cultures, organizations, social institutions, and processes that people develop.",
            'Chef': "Prepares food in restaurants or other food service establishments. Creates recipes, plans menus, and directs food preparation activities.",
            'Actor' : "Performs in plays, films, television, or other theatrical productions. Interprets scripts and portrays characters to entertain or inform audiences.",
            'Actuary': "Analyzes financial risks using mathematics, statistics, and financial theory. Often works in insurance, pensions, and other risk management fields.",
            'Aerospace Engineer': "Designs, tests, and manages the production of aircraft, spacecraft, satellites, and missiles. Works in both commercial and defense sectors.",
            'Anthropologist': "Studies human societies, cultures, and their development. May specialize in archaeology, biological, cultural, or linguistic anthropology.",
            'Arts Administrator': "Manages operations of arts organizations such as theaters, galleries, or music venues. Coordinates events, budgets, and community outreach.",
            'Automotive Engineer': "Designs and develops vehicle systems including engines, chassis, and electronics. Works on improving performance, safety, and efficiency.",
            'Broadcast Journalist': "Researches, writes, and delivers news stories for television, radio, or online media. Often works in fast-paced and deadline-driven environments.",
            'Business Intelligence Analyst': "Uses data analytics and visualization tools to help businesses make strategic decisions. Develops reports and dashboards for key stakeholders.",
            'Catering Manager': "Oversees food service operations for events or institutions. Plans menus, manages staff, and ensures high-quality service and food safety.",
            'Chemical Process Engineer': "Designs and optimizes processes for large-scale manufacturing. Focuses on transforming raw materials into valuable products efficiently.",
            'Choreographer': "Creates and arranges dance routines for performances in theater, film, music videos, or live events. Coaches dancers and refines performances.",
            'Communications Consultant': "Advises organizations on effective communication strategies. Helps improve public relations, branding, internal messaging, and media outreach.",
            'Community Development Manager': "Works with local governments or NGOs to improve community services and resources. Engages stakeholders to implement development projects.",
            'Composer': "Writes and arranges original music for films, orchestras, games, or digital media. May work independently or collaborate with performers and producers.",
            'Content Creator': "Produces digital content such as videos, blogs, or social media posts to entertain, educate, or market products. Often works as an influencer or brand partner.",
            'Copywriter': "Crafts persuasive and engaging written content for advertisements, websites, social media, and product descriptions. Focuses on brand tone and messaging.",
            'Corporate Trainer': "Designs and delivers professional development programs for employees. Covers skills such as communication, leadership, or technical proficiencies.",
            'Corporate Treasurer': "Manages an organization’s financial health, especially cash flow, investments, and risk. Ensures liquidity and handles financial planning.",
            'Credit Analyst': "Evaluates creditworthiness of individuals or businesses by assessing financial data. Works for banks, investment firms, or credit rating agencies.",
            'Cryptocurrency Analyst': "Studies digital currencies, blockchain trends, and market behaviors. Provides investment insights or risk evaluations for firms or individual investors.",
            'Cultural Researcher': "Studies traditions, customs, and societal behaviors within communities. Often works in academia, policy, or market research to inform cross-cultural understanding.",
            'Curriculum Developer': "Designs educational programs and learning materials for schools, training centers, or online platforms. Ensures content meets learning objectives and standards.",
            'Dancer': "Performs choreographed routines for stage, television, or film. May specialize in genres such as ballet, contemporary, hip-hop, or ballroom.",
            'Economist': "Analyzes economic trends, develops forecasts, and advises on policies or business strategies. Works with data models to understand market behavior.",
            'Educational Consultant': "Advises schools, parents, or organizations on academic planning, curriculum choices, and education policy. May also support student learning plans.",
            'Educational Psychologist': "Studies how people learn and develop educational interventions. Often works in schools to support students with learning difficulties.",
            'Electrical Engineer': "Designs, develops, and tests electrical systems and equipment such as circuits, motors, or communication systems. Applies principles of electricity and electronics.",
            'Environmental Engineer': "Develops solutions to environmental problems, including pollution control, waste management, and sustainable infrastructure.",
            'Event Planner': "Coordinates all aspects of events such as weddings, corporate functions, or festivals. Manages logistics, budgeting, and vendor relations.",
            'Financial Planner': "Helps individuals and businesses manage their finances, including investments, retirement planning, insurance, and estate planning.",
            'Food Critic': "Evaluates and writes reviews on restaurants, chefs, and food trends. Often works for magazines, blogs, or media outlets and influences public opinion.",
            'Food Scientist': "Studies the physical, biological, and chemical makeup of food. Works to improve food safety, preservation, nutrition, and processing techniques.",
            'Fundraising Manager': "Plans and executes campaigns to raise money for nonprofits, charities, or institutions. Builds relationships with donors and manages grant applications.",
            'Hospitality Manager': "Oversees operations in hotels, resorts, or hospitality venues. Ensures customer satisfaction, manages staff, and handles budgets and services.",
            'Human Resources Manager': "Directs HR policies including recruitment, employee relations, benefits, and compliance. Ensures a productive and legally sound workplace.",
            'Human Rights Advocate': "Works to protect and promote civil and human rights. May engage in research, policy work, education, or grassroots activism.",
            'International Relations Specialist': "Analyzes global political and economic issues to advise governments, NGOs, or corporations. May focus on diplomacy, policy, or security.",
            'Investment Banker': "Assists organizations in raising capital, mergers, and acquisitions. Analyzes financial data and markets to advise on strategic financial decisions.",
            'Management Consultant': "Advises companies on how to improve efficiency, profitability, and operations. Works across industries to solve complex business problems.",
            'Mechanical Engineer': "Designs and builds mechanical systems such as engines, tools, and machines. Applies principles of physics and materials science.",
            'Media Strategist': "Develops media plans and advertising strategies to reach target audiences. Works with various platforms like television, digital, print, and social media.",
            'Museum Educator': "Designs educational programs and exhibits for museums or galleries. Engages visitors through interactive activities and guided tours to enhance learning.",
            'Nutritional Consultant': "Advises individuals or organizations on healthy eating habits, nutritional plans, and wellness strategies. Often works in healthcare or wellness sectors.",
            'Online Learning Specialist': "Develops and manages digital courses and educational programs. Ensures content is engaging and accessible for online learners.",
            'Opera Singer': "Performs classical music in operas or concerts. Requires advanced vocal training and often specializes in a specific style or repertoire.",
            'Pastry Chef': "Specializes in baking pastries, cakes, and other desserts. Works in bakeries, restaurants, or for special events, combining creativity with technical skill.",
            'Podcast Producer': "Manages the production of podcasts, including planning content, editing audio, and coordinating with hosts and guests. Ensures high-quality sound and storytelling.",
            'Policy Analyst': "Analyzes and evaluates the impact of policies and regulations. Works in government or think tanks to provide insights for decision-making and policy development.",
            'Political Scientist': "Studies political systems, behaviors, and policies. Conducts research on governance, elections, and international relations to inform public opinion or policy.",
            'Product Manager': "Oversees the development and lifecycle of a product. Works with cross-functional teams to design, test, and launch new products in response to market needs.",
            'Professional Musician': "Performs music professionally as a soloist or part of a group. May work in recording, live performance, teaching, or composition across genres.",
            'Public Relations Specialist': "Manages an organization’s image and communications with the public. Writes press releases, handles media inquiries, and manages crisis communication.",
            'Real Estate Appraiser': "Assesses the market value of properties for sale, tax assessment, mortgage, or investment purposes. Uses local market data and inspection reports.",
            'Restaurant Manager': "Oversees daily operations of a restaurant including staff management, customer service, inventory, and financial performance.",
            'Risk Manager': "Identifies, assesses, and mitigates financial or operational risks within an organization. Develops strategies to reduce exposure to losses or regulatory issues.",
            'Robotics Engineer': "Designs, builds, and tests robots or robotic systems. Applies knowledge of mechanical, electrical, and software engineering to automation challenges.",
            'School Administrator': "Manages operations in educational institutions, including curriculum planning, staff supervision, and compliance with policies and regulations.",
            'Securities Trader': "Buys and sells stocks, bonds, or other financial instruments on behalf of clients or firms. Works in fast-paced environments like stock exchanges or trading desks.",
            'Social Media Manager': "Develops and executes strategies to grow an organization’s presence on social platforms. Creates content, monitors engagement, and analyzes metrics.",
            'Social Worker': "Supports individuals and families facing challenges such as poverty, abuse, or mental health issues. Connects them to resources and advocates for their well-being.",
            'Software Engineering Manager': "Leads a team of software developers, oversees project timelines, ensures code quality, and aligns technical efforts with business goals.",
            'Sommelier': "A wine expert responsible for wine selection, pairing, and service in fine dining settings. Possesses deep knowledge of wine regions, varieties, and tasting techniques.",
            'Sound Engineer': "Manages audio quality during live events or recordings. Operates soundboards, mixes tracks, and ensures clarity and balance in sound production.",
            'Special Education Specialist': "Works with students who have learning disabilities or special needs. Develops individualized education plans and adapts teaching methods.",
            'Stage Designer': "Designs sets and scenery for theater, film, or television. Collaborates with directors and lighting teams to create visual environments for performances.",
            'Startup Founder': "Launches and manages a new business venture. Responsible for product development, fundraising, team building, and navigating market challenges.",
            'Structural Engineer': "Designs and analyzes the framework of buildings, bridges, and infrastructure to ensure safety and stability under various conditions.",
            'Supply Chain Manager': "Oversees the entire lifecycle of a product, from sourcing materials to delivery. Aims to improve efficiency, reduce costs, and ensure timely production.",
            'Technical Writer': "Creates user manuals, product documentation, and guides for technical subjects. Translates complex information into clear, accessible content.",
            'Theatre Director': "Interprets scripts and oversees all artistic aspects of a stage production. Guides actors, stage crew, and designers to bring the vision to life.",
            'University Professor': "Teaches courses at the college or university level and conducts academic research in their field. May also publish scholarly work and advise students.",
            'Urban Planner': "Develops plans and policies for land use in cities and communities. Works to improve infrastructure, transportation, housing, and sustainability."
        

            # Note: In a complete implementation, you would add descriptions for all 120 careers in the dataset
        }
    
    def get_career_resources(self):
        """Dictionary of resources for pursuing careers"""
        return {
            # Technology resources
            'Software Engineer': [
                "Degree programs: Computer Science, Software Engineering",
                "Certifications: AWS Certified Developer, Microsoft Certified: Azure Developer",
                "Online platforms: Codecademy, LeetCode, GitHub",
                "Professional organizations: IEEE Computer Society, ACM"
            ],
            'Data Scientist': [
                "Degree programs: Data Science, Statistics, Computer Science",
                "Certifications: IBM Data Science Professional, Google Data Analytics",
                "Online platforms: Kaggle, DataCamp, Coursera",
                "Professional organizations: Data Science Association, INFORMS"
            ],
            'Cybersecurity Analyst': [
                "Degree programs: Cybersecurity, Information Technology",
                "Certifications: CompTIA Security+, Certified Information Systems Security Professional (CISSP)",
                "Online platforms: TryHackMe, HackTheBox, Cybrary",
                "Professional organizations: (ISC)², ISACA"
            ],
            
            # Healthcare resources
            'Doctor/Physician': [
                "Degree programs: Pre-medicine undergraduate, followed by Medical Doctor (MD) or Doctor of Osteopathic Medicine (DO)",
                "Examinations: MCAT, USMLE (for MDs), COMLEX (for DOs)",
                "Resources: AAMC, MedEdPORTAL, UpToDate",
                "Professional organizations: American Medical Association, specialty-specific organizations"
            ],
            'Psychologist': [
                "Degree programs: Psychology (PhD or PsyD for clinical practice)",
                "Certifications: State licensure required for clinical practice",
                "Resources: American Psychological Association, PsycINFO",
                "Professional organizations: American Psychological Association, Association for Psychological Science"
            ],
            
            # Creative resources
            'Graphic Designer': [
                "Degree programs: Graphic Design, Visual Communications",
                "Software skills: Adobe Creative Suite (Photoshop, Illustrator, InDesign)",
                "Portfolio platforms: Behance, Dribbble",
                "Professional organizations: AIGA (American Institute of Graphic Arts)"
            ],
            'Writer/Author': [
                "Degree programs: Creative Writing, English, Journalism",
                "Resources: Masterclass, Writer's Digest, Reedsy",
                "Communities: NaNoWriMo, Critique Circle",
                "Professional organizations: Authors Guild, Society of Children's Book Writers and Illustrators"
            ],
            
            # Business resources
            'Financial Analyst': [
                "Degree programs: Finance, Accounting, Economics",
                "Certifications: CFA (Chartered Financial Analyst), FRM (Financial Risk Manager)",
                "Resources: Bloomberg Terminal, Wall Street Journal, Financial Times",
                "Professional organizations: CFA Institute, Global Association of Risk Professionals"
            ],
            'Entrepreneur': [
                "Degree programs: Business Administration, Entrepreneurship",
                "Resources: Y Combinator Startup School, Entrepreneurship magazines (Inc., Entrepreneur)",
                "Networks: Startup incubators, local entrepreneur meetups",
                "Organizations: Entrepreneurs' Organization, Young Entrepreneurs Council"
            ],
            
            # Adding more examples...
            'Teacher': [
                "Degree programs: Education, subject-specific degrees with teaching certification",
                "Certifications: State teaching license, National Board Certification",
                "Resources: Edutopia, Khan Academy Teacher Resources",
                "Professional organizations: National Education Association, subject-specific teacher associations"
            ],
            'Civil Engineer': [
                "Degree programs: Civil Engineering",
                "Certifications: Professional Engineer (PE) license",
                "Resources: AutoCAD, Civil engineering handbooks",
                "Professional organizations: American Society of Civil Engineers"

            ],
            'Actuary': [
                "Degree programs: Actuarial Science, Mathematics, Statistics",
                "Certifications: SOA or CAS actuarial exams",
                "Tools: Excel, R, Python, actuarial software",
                "Resources: BeAnActuary.org, SOA study materials",
                "Organizations: Society of Actuaries, Casualty Actuarial Society"
            ],
            'Aerospace Engineer': [
                "Degree programs: Aerospace Engineering",
                "Certifications: FE, PE",
                "Tools: MATLAB, CATIA, SolidWorks, ANSYS",
                "Platforms: NASA Academy, edX Aerospace Courses",
                "Organizations: AIAA (American Institute of Aeronautics and Astronautics)"
            ],
            'Anthropologist': [
                "Degree programs: Anthropology, Sociology",
                "Resources: JSTOR, AnthroSource",
                "Tools: Ethnographic software, recording tools",
                "Organizations: American Anthropological Association, Society for Applied Anthropology"
            ],
            'Arts Administrator': [
                "Degree programs: Arts Administration, Arts Management",
                "Certifications: Nonprofit Management Certificates",
                "Resources: Americans for the Arts, National Arts Strategies",
                "Tools: CRM and ticketing platforms (e.g., Tessitura, Salesforce)"
            ],
            'Automotive Engineer': [
                "Degree programs: Mechanical or Automotive Engineering",
                "Certifications: FE, PE, automotive-specific training (SAE)",
                "Tools: CAD software, MATLAB, Simulink",
                "Organizations: SAE International, ASME"
            ],
            'Broadcast Journalist': [
                "Degree programs: Journalism, Communications",
                "Tools: Adobe Premiere, newsroom software, teleprompters",
                "Resources: Poynter Institute, RTDNA",
                "Organizations: National Association of Broadcasters, SPJ"
            ],
            'Business Intelligence Analyst': [
                "Degree programs: Business Analytics, Information Systems",
                "Certifications: Tableau Specialist, Microsoft Power BI",
                "Tools: SQL, Power BI, Tableau, Excel",
                "Resources: Udacity, IBM Analytics Academy",
                "Organizations: TDWI (Transforming Data with Intelligence)"
            ],
            'Catering Manager': [
                "Degree programs: Hospitality Management, Culinary Arts",
                "Certifications: ServSafe, Certified Catering Professional",
                "Resources: Catersource, The Knot Pro",
                "Tools: Event planning software, POS systems"
            ],
            'Chemical Process Engineer': [
                "Degree programs: Chemical Engineering",
                "Certifications: FE, PE",
                "Tools: Aspen HYSYS, ChemCAD, MATLAB",
                "Organizations: AIChE (American Institute of Chemical Engineers)"
            ],
            'Choreographer': [
                "Degree programs: Dance, Performing Arts",
                "Resources: Dance Magazine, MasterClass",
                "Tools: Music editing software, rehearsal planning tools",
                "Organizations: Dance/USA, National Dance Education Organization"
            ],
            'Communications Consultant': [
                "Degree programs: Communications, Public Relations",
                "Certifications: PRSA Accreditation, Strategic Communications Certificates",
                "Tools: Canva, Google Workspace, media monitoring tools",
                "Resources: PR News, Institute for Public Relations",
                "Organizations: Public Relations Society of America (PRSA)"
            ],
            'Community Development Manager': [
                "Degree programs: Urban Planning, Public Administration, Social Work",
                "Resources: Community Tool Box, Urban Institute",
                "Tools: GIS software, stakeholder mapping tools",
                "Organizations: American Planning Association, IEDC"
            ],
            'Composer': [
                "Degree programs: Music Composition, Music Theory",
                "Tools: Sibelius, Finale, DAWs (Logic Pro, Ableton)",
                "Resources: Berklee Online, Score Exchange",
                "Organizations: American Composers Forum, ASCAP"
            ],
            'Content Creator': [
                "Degree programs: Digital Media, Communications, Marketing",
                "Platforms: YouTube Creator Academy, Skillshare",
                "Tools: Adobe Creative Suite, Final Cut Pro, Canva",
                "Communities: Creator Economy Groups, Patreon, Discord Creators"
            ],
            'Copywriter': [
                "Degree programs: English, Marketing, Communications",
                "Certifications: Copywriting certifications (e.g., AWAI)",
                "Resources: Copyhackers, HubSpot Academy",
                "Tools: Grammarly, Hemingway Editor, Notion",
                "Communities: ProCopywriters, Freelance Writers Den"
            ],
            'Corporate Trainer': [
                "Degree programs: Human Resource Development, Organizational Psychology",
                "Certifications: ATD Certification, SHRM Learning",
                "Tools: Learning Management Systems (LMS), presentation software",
                "Organizations: ATD (Association for Talent Development)"
            ],
            'Corporate Treasurer': [
                "Degree programs: Finance, Accounting",
                "Certifications: CTP (Certified Treasury Professional)",
                "Tools: Treasury management software, Excel, Bloomberg",
                "Organizations: AFP (Association for Financial Professionals)"
            ],
            'Credit Analyst': [
                "Degree programs: Finance, Economics, Business",
                "Certifications: CFA, Credit Risk Certification",
                "Tools: Moody’s, Excel, SQL",
                "Resources: Fitch Ratings, Investopedia",
                "Organizations: Risk Management Association (RMA)"
            ],
            'Cryptocurrency Analyst': [
                "Degree programs: Finance, Computer Science",
                "Certifications: Blockchain Council Certifications, Certified Crypto Analyst",
                "Tools: CoinMarketCap, TradingView, crypto wallets",
                "Resources: CoinDesk, Decrypt, Binance Academy"
            ],
            'Cultural Researcher': [
                "Degree programs: Cultural Studies, Anthropology, Sociology",
                "Tools: NVivo, qualitative data analysis tools",
                "Resources: Culture Unbound, Taylor & Francis Online",
                "Organizations: International Cultural Research Network"
            ],
            'Curriculum Developer': [
                "Degree programs: Education, Curriculum and Instruction",
                "Certifications: Instructional Design Certificates, Google for Education",
                "Tools: LMS platforms, instructional design software",
                "Resources: ASCD, Edutopia"
            ],
            'Dancer': [
                "Degree programs: Dance, Performing Arts",
                "Resources: Dance Spirit, Online Dance Academy",
                "Tools: Dance mirrors, rehearsal software",
                "Organizations: Dance/USA, National Dance Education Organization"
            ],
            'Economist': [
                "Degree programs: Economics, Statistics, Public Policy",
                "Certifications: Economic Modeling Training (EMT)",
                "Resources: World Bank Data, IMF eLibrary, NBER",
                "Tools: Stata, R, EViews",
                "Organizations: American Economic Association"
            ],
            'Educational Consultant': [
                "Degree programs: Education, Counseling, Educational Leadership",
                "Certifications: Teaching license, Educational Consultant Certificate",
                "Resources: ASCD, Education Week",
                "Tools: Student assessment tools, LMS platforms",
                "Organizations: Independent Educational Consultants Association (IECA)"
            ],
            'Educational Psychologist': [
                "Degree programs: Educational Psychology, School Psychology",
                "Certifications: Licensed Psychologist, NCSP (Nationally Certified School Psychologist)",
                "Resources: ERIC, APA Division 15",
                "Tools: Psychometric tools, behavioral assessment software",
                "Organizations: APA, NASP"
            ],
            'Electrical Engineer': [
                "Degree programs: Electrical Engineering",
                "Certifications: FE, PE",
                "Tools: MATLAB, Multisim, AutoCAD Electrical",
                "Resources: IEEE Xplore, Coursera Electrical Engineering",
                "Organizations: IEEE (Institute of Electrical and Electronics Engineers)"
            ],
            'Environmental Engineer': [
                "Degree programs: Environmental Engineering, Civil Engineering",
                "Certifications: FE, PE, LEED Accreditation",
                "Tools: GIS software, AutoCAD, environmental modeling tools",
                "Resources: EPA Training Center, ASCE publications",
                "Organizations: American Academy of Environmental Engineers"
            ],
            'Event Planner': [
                "Degree programs: Hospitality Management, Event Management",
                "Certifications: Certified Meeting Professional (CMP), CSEP",
                "Tools: Eventbrite, Asana, project management tools",
                "Resources: MeetingsNet, EventMB",
                "Organizations: Meeting Professionals International (MPI)"
            ],
            'Financial Planner': [
                "Degree programs: Finance, Economics, Business",
                "Certifications: CFP (Certified Financial Planner), CPA",
                "Tools: QuickBooks, Excel, planning software",
                "Resources: CFP Board, SmartAsset",
                "Organizations: Financial Planning Association (FPA)"
            ],
            'Food Critic': [
                "Degree programs: Journalism, Culinary Arts",
                "Resources: Eater, Bon Appétit, Zagat",
                "Tools: Content management systems, social media platforms",
                "Organizations: Association of Food Journalists (AFJ)"
            ],
            'Food Scientist': [
                "Degree programs: Food Science, Nutrition, Chemistry",
                "Certifications: Certified Food Scientist (CFS)",
                "Tools: Lab equipment, SPSS, chromatography systems",
                "Resources: Institute of Food Technologists, Food Quality Magazine",
                "Organizations: IFT (Institute of Food Technologists)"
            ],
            'Fundraising Manager': [
                "Degree programs: Nonprofit Management, Communications",
                "Certifications: CFRE (Certified Fund Raising Executive)",
                "Tools: CRM systems (e.g., DonorPerfect, Raiser’s Edge)",
                "Resources: Nonprofit Quarterly, AFP Fundraising Guides",
                "Organizations: Association of Fundraising Professionals (AFP)"
            ],
            'Hospitality Manager': [
                "Degree programs: Hospitality Management, Business Administration",
                "Certifications: CHA (Certified Hotel Administrator)",
                "Tools: Hotel property management systems (PMS), POS systems",
                "Resources: Hospitality Net, Cornell Hotel School resources",
                "Organizations: AHLA (American Hotel & Lodging Association)"
            ],
            'Human Resources Manager': [
                "Degree programs: Human Resources, Business Management",
                "Certifications: SHRM-CP, PHR/SPHR",
                "Tools: HRIS systems, performance management tools",
                "Resources: SHRM.org, HR Dive",
                "Organizations: Society for Human Resource Management (SHRM)"
            ],
            'Human Rights Advocate': [
                "Degree programs: International Relations, Law, Human Rights",
                "Certifications: Human Rights Education courses (Amnesty, UN)",
                "Resources: Human Rights Watch, UN OHCHR, Amnesty International",
                "Tools: Policy research tools, communication platforms",
                "Organizations: Amnesty International, Human Rights First"
            ],
            'International Relations Specialist': [
                "Degree programs: International Relations, Political Science",
                "Certifications: Foreign Service Officer Test (FSOT), Peace and Conflict certifications",
                "Resources: Foreign Policy, UN Careers, Council on Foreign Relations",
                "Organizations: APSIA, UNA-USA"
            ],
            'Investment Banker': [
                "Degree programs: Finance, Business, Economics",
                "Certifications: FINRA Series 7 & 63, CFA",
                "Tools: Bloomberg Terminal, Excel, PitchBook",
                "Resources: Wall Street Oasis, Mergers & Inquisitions",
                "Organizations: SIFMA, CFA Institute"
            ],
            'Management Consultant': [
                "Degree programs: Business Administration, Management",
                "Certifications: Certified Management Consultant (CMC)",
                "Resources: McKinsey Insights, Harvard Business Review",
                "Tools: PowerPoint, Excel, Miro",
                "Organizations: Institute of Management Consultants (IMC USA)"
            ],
            'Mechanical Engineer': [
                "Degree programs: Mechanical Engineering",
                "Certifications: FE, PE",
                "Tools: SolidWorks, AutoCAD, MATLAB",
                "Resources: EngineeringToolBox, Coursera",
                "Organizations: ASME (American Society of Mechanical Engineers)"
            ],
            'Media Strategist': [
                "Degree programs: Marketing, Communications, Media Planning",
                "Certifications: Google Ads, Facebook Blueprint",
                "Tools: Google Analytics, Hootsuite, SEMrush",
                "Resources: Adweek, Think with Google",
                "Organizations: American Marketing Association"
            ],
            'Museum Educator': [
                "Degree programs: Museum Studies, Education, Art History",
                "Certifications: Teaching credential or Museum Education Certificate",
                "Resources: Smithsonian Learning Lab, Museum-Ed",
                "Tools: Exhibit planning software, virtual museum tools",
                "Organizations: American Alliance of Museums"
            ],
            'Nutritional Consultant': [
                "Degree programs: Nutrition, Dietetics, Public Health",
                "Certifications: Certified Nutrition Specialist (CNS), RD",
                "Resources: Academy of Nutrition and Dietetics, Precision Nutrition",
                "Tools: Diet planning software, health tracking apps",
                "Organizations: American Nutrition Association"
            ],
            'Online Learning Specialist': [
                "Degree programs: Instructional Design, Educational Technology",
                "Certifications: Google Certified Educator, Adobe Captivate",
                "Resources: eLearning Industry, Coursera Instructional Design",
                "Tools: Articulate, Canvas, Moodle",
                "Organizations: International Society for Technology in Education (ISTE)"
            ],
            'Opera Singer': [
                "Degree programs: Vocal Performance, Music",
                "Certifications: Professional vocal training, conservatory diplomas",
                "Resources: Classical Singer Magazine, OperaWire",
                "Tools: Vocal warm-up tools, recording software",
                "Organizations: National Association of Teachers of Singing (NATS)"
            ],
            'Pastry Chef': [
                "Degree programs: Baking and Pastry Arts, Culinary Arts",
                "Certifications: Certified Pastry Culinarian (CPC)",
                "Resources: The French Pastry School, Craftsy",
                "Tools: Pastry equipment, recipe costing tools",
                "Organizations: American Culinary Federation"
            ],
            'Podcast Producer': [
                "Degree programs: Audio Production, Journalism, Media Studies",
                "Certifications: Podcast Production Courses (Coursera, Udemy)",
                "Tools: Adobe Audition, Audacity, Descript",
                "Resources: Podcast Movement, Podnews",
                "Organizations: Association of Independents in Radio (AIR)"
            ],
            'Policy Analyst': [
                "Degree programs: Public Policy, Political Science, Economics",
                "Certifications: Policy Analysis Training (Brookings, LSE)",
                "Tools: STATA, Excel, policy simulation tools",
                "Resources: RAND Corporation, PolicyLink",
                "Organizations: American Political Science Association (APSA)"
            ],
            'Political Scientist': [
                "Degree programs: Political Science, International Relations",
                "Certifications: Academic credentials (PhD for research roles)",
                "Resources: APSA, Political Studies Association",
                "Tools: Statistical analysis software (SPSS, R)",
                "Organizations: American Political Science Association"
            ],
            'Product Manager': [
                "Degree programs: Business, Computer Science, UX Design",
                "Certifications: Product Management Certificate (PMI, General Assembly)",
                "Tools: Jira, Confluence, Figma, Trello",
                "Resources: Product School, Mind the Product",
                "Organizations: Product Development and Management Association (PDMA)"
            ],
            'Professional Musician': [
                "Degree programs: Music Performance, Composition",
                "Certifications: Music certifications (ABRSM, Trinity)",
                "Resources: Berklee Online, Soundfly, Performer's Guide",
                "Tools: Sheet music apps, DAWs, recording equipment",
                "Organizations: American Federation of Musicians"
            ],
            'Public Relations Specialist': [
                "Degree programs: Public Relations, Communications",
                "Certifications: APR (Accredited in Public Relations)",
                "Resources: PRSA, Ragan Communications",
                "Tools: Cision, Meltwater, media list databases",
                "Organizations: Public Relations Society of America (PRSA)"
            ],
            'Real Estate Appraiser': [
                "Degree programs: Real Estate, Finance, Economics",
                "Certifications: State Appraiser License, AQB Certified",
                "Tools: MLS, appraisal software, market data tools",
                "Resources: Appraisal Institute, Real Estate Appraiser Directory",
                "Organizations: National Association of Realtors (NAR)"
            ],
            'Restaurant Manager': [
                "Degree programs: Hospitality Management, Restaurant Management",
                "Certifications: ServSafe Manager Certification",
                "Tools: POS systems, inventory and scheduling software",
                "Resources: National Restaurant Association, Restaurant Business Online",
                "Organizations: National Restaurant Association"
            ],
            'Risk Manager': [
                "Degree programs: Risk Management, Finance, Business",
                "Certifications: CRM (Certified Risk Manager), FRM",
                "Tools: Risk analysis software, dashboards, audit tools",
                "Resources: RIMS Risk Management Magazine, PRMIA",
                "Organizations: Risk Management Society (RIMS)"
            ],
            'Robotics Engineer': [
                "Degree programs: Robotics, Mechatronics, Mechanical Engineering",
                "Certifications: Robotics Certification Standards (RIA, ABB)",
                "Tools: ROS, Python, CAD, Arduino",
                "Resources: MIT OpenCourseWare, Udacity Robotics",
                "Organizations: IEEE Robotics and Automation Society"
            ],
            'School Administrator': [
                "Degree programs: Educational Leadership, School Administration",
                "Certifications: School Principal Certification, Ed Leadership Licensure",
                "Resources: ASCD, NAESP, Education Corner",
                "Tools: SIS systems, school evaluation platforms",
                "Organizations: National Association of Elementary School Principals (NAESP)"
            ],
            'Securities Trader': [
                "Degree programs: Finance, Economics, Business",
                "Certifications: FINRA Series 7, 63, 57",
                "Tools: Bloomberg Terminal, trading platforms, Excel",
                "Resources: Investopedia, Wall Street Journal",
                "Organizations: SIFMA, CFA Institute"
            ],
            'Social Media Manager': [
                "Degree programs: Marketing, Communications, Digital Media",
                "Certifications: Meta Blueprint, Hootsuite Academy",
                "Tools: Buffer, Canva, Google Analytics",
                "Resources: Sprout Social, HubSpot Blog",
                "Organizations: Social Media Examiner"
            ],
            'Social Worker': [
                "Degree programs: Social Work (BSW, MSW)",
                "Certifications: Licensed Clinical Social Worker (LCSW)",
                "Resources: NASW Practice Tools, Social Work Helper",
                "Tools: Case management systems, intervention planning tools",
                "Organizations: National Association of Social Workers (NASW)"
            ],
            'Software Engineering Manager': [
                "Degree programs: Computer Science, Engineering Management",
                "Certifications: PMP, Scrum Master, AWS Certifications",
                "Tools: Jira, GitHub, Docker",
                "Resources: Engineering Manager's Handbook, LeadDev",
                "Organizations: ACM, IEEE Software"
            ],
            'Sommelier': [
                "Degree programs: Hospitality, Culinary Arts",
                "Certifications: Court of Master Sommeliers, WSET",
                "Resources: GuildSomm, Wine Folly",
                "Tools: Wine inventory software, tasting sheets",
                "Organizations: Court of Master Sommeliers, Society of Wine Educators"
            ],
            'Sound Engineer': [
                "Degree programs: Audio Engineering, Music Production",
                "Certifications: Avid Pro Tools Certification, AES courses",
                "Tools: DAWs (Pro Tools, Logic Pro), audio interfaces",
                "Resources: Sound on Sound, Mix Magazine",
                "Organizations: Audio Engineering Society (AES)"
            ],
            'Special Education Specialist': [
                "Degree programs: Special Education, Educational Psychology",
                "Certifications: State licensure, Board Certification",
                "Tools: IEP software, assessment tools",
                "Resources: Council for Exceptional Children (CEC), Edutopia",
                "Organizations: National Association of Special Education Teachers (NASET)"
            ],
            'Stage Designer': [
                "Degree programs: Theater Design, Scenic Design",
                "Certifications: USITT training programs",
                "Tools: SketchUp, AutoCAD, lighting design software",
                "Resources: Stage Directions, TheatreArtLife",
                "Organizations: United States Institute for Theatre Technology (USITT)"
            ],
            'Startup Founder': [
                "Degree programs: Entrepreneurship, Business Administration",
                "Certifications: Y Combinator Startup School, Lean Startup Training",
                "Tools: Pitch decks, CRMs, Notion, financial modeling tools",
                "Resources: Indie Hackers, TechCrunch, First Round Review",
                "Organizations: Entrepreneurs' Organization, Startup Grind"
            ],
            'Structural Engineer': [
                "Degree programs: Structural Engineering, Civil Engineering",
                "Certifications: PE, SE (Structural Engineer License)",
                "Tools: SAP2000, ETABS, AutoCAD",
                "Resources: Structure Magazine, AISC",
                "Organizations: SEI (Structural Engineering Institute)"
            ],
            'Supply Chain Manager': [
                "Degree programs: Supply Chain Management, Operations",
                "Certifications: APICS CPIM, CSCP",
                "Tools: SAP, Oracle SCM, logistics tracking systems",
                "Resources: Supply Chain Dive, Logistics Management",
                "Organizations: ASCM, Institute for Supply Management"
            ],
            'Technical Writer': [
                "Degree programs: Technical Communication, English, IT",
                "Certifications: STC Certification, Google Technical Writing Courses",
                "Tools: Markdown, FrameMaker, Snagit",
                "Resources: Write the Docs, TechWhirl",
                "Organizations: Society for Technical Communication (STC)"
            ],
            'Theatre Director': [
                "Degree programs: Theatre Arts, Directing",
                "Certifications: MFA in Directing, USITT Training",
                "Resources: American Theatre Magazine, StageMilk",
                "Tools: Script annotation software, rehearsal scheduling tools",
                "Organizations: Stage Directors and Choreographers Society (SDC)"
            ],
            'University Professor': [
                "Degree programs: PhD in relevant field",
                "Certifications: Teaching certification if required (depends on institution)",
                "Resources: Chronicle of Higher Education, JSTOR",
                "Tools: Research databases, LMS systems",
                "Organizations: American Association of University Professors (AAUP)"
            ],
            'Urban Planner': [
                "Degree programs: Urban Planning, Geography, Public Policy",
                "Certifications: AICP (American Institute of Certified Planners)",
                "Tools: GIS, AutoCAD, urban modeling software",
                "Resources: Planetizen, APA KnowledgeBase",
                "Organizations: American Planning Association (APA)"]
            
            # Note: In a complete implementation, you would add resources for all 120 careers in the dataset
        }
    
    def load_dataset(self):
        """Load the dataset and extract career categories"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully with {len(self.df)} samples.")
            
            # Get all unique career categories and specific careers
            self.career_categories = self.df.groupby('career_category')['specific_career'].unique().to_dict()
            print(f"Found {len(self.career_categories)} career categories.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self):
        """Create and fit preprocessing pipeline"""
        # Identify column types
        numeric_features = [
            'age', 'stem_interest_score', 'creativity_score', 
            'social_skills_score', 'analytical_skills_score', 
            'problem_solving_score', 'technical_aptitude_score', 
            'communication_score', 'annual_salary_expectation', 
            'job_growth_interest', 'leadership_potential_score',
            'work_life_balance_score', 'global_mobility_score'
        ]
        
        categorical_features = [
            'education_level', 'gender', 
            'career_category', 'work_environment_preference'
        ]
        
        # Create preprocessors
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Prepare features and target
        X = self.df.drop(['specific_career'], axis=1)
        y = self.df['specific_career']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit preprocessor
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names
        self.feature_names = numeric_features + list(
            self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        )
        
        # Fit model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_transformed, y_train)
        
        # Evaluate model
        X_test_transformed = self.preprocessor.transform(X_test)
        y_pred = self.model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        return X_test, y_test
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            print("Model not trained yet. Call preprocess_data() first.")
            return None
        
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, model_path='career_recommendation_model.pkl'):
        """Save the model and preprocessor"""
        if self.model is None:
            print("Model not trained yet. Call preprocess_data() first.")
            return
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'career_categories': self.career_categories
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='career_recommendation_model.pkl'):
        """Load the model and preprocessor"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.career_categories = model_data['career_categories']
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_career(self, user_data):
        """Predict career based on user data"""
        if self.model is None:
            print("Model not trained yet. Call preprocess_data() first.")
            return None
        
        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Transform user data
        user_transformed = self.preprocessor.transform(user_df)
        
        # Get probabilities for all careers
        career_probs = self.model.predict_proba(user_transformed)
        
        # Get indices of top 5 careers
        top_indices = career_probs[0].argsort()[-5:][::-1]
        
        # Get career names
        career_names = [self.model.classes_[i] for i in top_indices]
        
        # Get probabilities
        probabilities = [career_probs[0][i] for i in top_indices]
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'Career': career_names,
            'Match Score': [f"{prob:.1%}" for prob in probabilities]
        })
        
        return recommendations
    
    def get_user_input(self):
        """Interactive function to get user input"""
        print("\n===== Career Recommendation System =====")
        print("Please provide your information and preferences:")
        
        user_data = {}
        
        # Age
        while True:
            try:
                age = int(input("Enter your age (18-60): "))
                if 18 <= age <= 60:
                    user_data['age'] = age
                    break
                else:
                    print("Age must be between 20 and 50.")
            except ValueError:
                print("Please enter a valid age.")
        
        # Education level
        education_levels = ['High School', 'Associate Degree', 'Bachelor\'s', 'Master\'s', 'Professional Degree', 'PhD']
        print("\nEducation Levels:")
        for i, level in enumerate(education_levels, 1):
            print(f"{i}. {level}")
        
        while True:
            try:
                edu_choice = int(input("Select your education level (1-6): "))
                if 1 <= edu_choice <= 6:
                    user_data['education_level'] = education_levels[edu_choice-1]
                    break
                else:
                    print("Please select a number between 1 and 6.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Gender
        gender_options = ['Male', 'Female', 'Non-Binary', 'Prefer Not to Say']
        print("\nGender:")
        for i, gender in enumerate(gender_options, 1):
            print(f"{i}. {gender}")
        
        while True:
            try:
                gender_choice = int(input("Select your gender (1-4): "))
                if 1 <= gender_choice <= 4:
                    user_data['gender'] = gender_options[gender_choice-1]
                    break
                else:
                    print("Please select a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Skill scores
        skill_descriptions = {
            'stem_interest_score': "STEM (Science, Technology, Engineering, Math) interest",
            'creativity_score': "Creativity",
            'social_skills_score': "Social skills",
            'analytical_skills_score': "Analytical skills",
            'problem_solving_score': "Problem solving",
            'technical_aptitude_score': "Technical aptitude",
            'communication_score': "Communication",
            'leadership_potential_score': "Leadership potential",
            'work_life_balance_score': "Work-life balance importance",
            'global_mobility_score': "Global mobility interest (willingness to relocate)"
        }
        
        print("\nFor the following skills and preferences, rate yourself from 0-100:")
        print("(0 = Very Low, 50 = Average, 100 = Very High)")
        
        for skill, description in skill_descriptions.items():
            while True:
                try:
                    score = int(input(f"Rate your {description} (0-100): "))
                    if 0 <= score <= 100:
                        user_data[skill] = score
                        break
                    else:
                        print("Score must be between 0 and 100.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # Career category preference
        print("\nCareer Categories:")
        categories = list(self.career_categories.keys())
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            try:
                category_choice = int(input(f"Select your preferred career category (1-{len(categories)}): "))
                if 1 <= category_choice <= len(categories):
                    user_data['career_category'] = categories[category_choice-1]
                    break
                else:
                    print(f"Please select a number between 1 and {len(categories)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Work environment preference
        work_environments = ['Remote', 'Hybrid', 'On-site', 'Flexible', 'Travel-based']
        print("\nWork Environment Preferences:")
        for i, env in enumerate(work_environments, 1):
            print(f"{i}. {env}")
        
        while True:
            try:
                env_choice = int(input("Select your preferred work environment (1-5): "))
                if 1 <= env_choice <= 5:
                    user_data['work_environment_preference'] = work_environments[env_choice-1]
                    break
                else:
                    print("Please select a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Salary expectation (MODIFIED: now takes a range)
        print("\nSalary Expectation Range:")
        while True:
            try:
                min_salary = int(input("Enter your minimum expected annual salary (USD): "))
                max_salary = int(input("Enter your maximum expected annual salary (USD): "))
                
                if min_salary >= 0 and max_salary >= min_salary:
                    # Use the average of the range for the model
                    user_data['annual_salary_expectation'] = (min_salary + max_salary) // 2
                    break
                else:
                    if min_salary <= 0:
                        print("Minimum salary must be positive.")
                    else:
                        print("Maximum salary must be greater than or equal to minimum salary.")
            except ValueError:
                print("Please enter valid numbers.")
        
        # Job growth interest
        while True:
            try:
                growth = int(input("Rate your interest in job growth opportunities (0-100): "))
                if 0 <= growth <= 100:
                    user_data['job_growth_interest'] = growth
                    break
                else:
                    print("Score must be between 0 and 100.")
            except ValueError:
                print("Please enter a valid number.")
        
        return user_data
    
    def display_career_info(self, career):
        """Display detailed information about a career"""
        print(f"\n===== {career} =====")
        
        # Find category
        category = None
        for cat, careers in self.career_categories.items():
            if career in careers:
                category = cat
                break
        
        if category:
            print(f"Category: {category}")
        
        # Description
        if career in self.career_descriptions:
            print(f"\nDescription: {self.career_descriptions[career]}")
        else:
            print("\nDescription: Information not available")
        
        # Resources for pursuing this career
        if career in self.career_resources:
            print("\nHow to Pursue This Career:")
            for resource in self.career_resources[career]:
                print(f"- {resource}")
        else:
            print("\nCareer Path Information:")
            print("- Research relevant degree programs in this field")
            print("- Look for professional certifications that can enhance your credentials")
            print("- Join professional organizations related to this field")
            print("- Build a portfolio or gain experience through internships or entry-level positions")


def main():
    # Create recommendation system
    recommender = CareerRecommendationSystem()
    
    # Check if model exists
    import os
    model_path = 'career_recommendation_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        recommender.load_model(model_path)
    else:
        # Train new model
        print("Training new model...")
        recommender.preprocess_data()
        
        # Show feature importance
        importance = recommender.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))
        
        # Save model
        recommender.save_model(model_path)
    
    # Get user input
    user_data = recommender.get_user_input()
    
    # Get recommendations
    recommendations = recommender.predict_career(user_data)
    
    # Show recommendations
    print("\n===== Career Recommendations =====")
    print(recommendations)
    
    # Display detailed information for each recommended career
    print("\n===== Detailed Career Information =====")
    for career in recommendations['Career']:
        recommender.display_career_info(career)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()