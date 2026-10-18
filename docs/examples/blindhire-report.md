# Screening report — Senior AI Engineer

Rubric weights: {'skills_match': 0.4, 'relevant_experience': 0.3, 'delivery_impact': 0.2, 'communication': 0.1} · threshold 3.2 · shortlist size 3

| Rank | Candidate | Weighted /5 | Must-haves | Decision |
|---|---|---|---|---|
| 1 | C-01 (Anika Rao) | 5.00 | all met | Interview |
| 2 | C-02 (Li Wei) | 4.30 | all met | Interview |
| 3 | C-04 (Sara Khan) | 2.60 | missing: Python in production for 3+ years | Decline |
| 4 | C-05 (Tomás Herrera) | 1.90 | missing: Python in production for 3+ years; Built and shipped an LLM application (RAG, agents or similar) to real users; Deployed and operated services on a public cloud (AWS, GCP or Azure) | Decline |
| 5 | C-03 (Omar Haddad) | 1.10 | missing: Python in production for 3+ years; Built and shipped an LLM application (RAG, agents or similar) to real users | Decline |

_Scores were produced from anonymised CVs. Names were attached after the shortlist was approved._

## C-01 — Anika Rao

The candidate exceeds the senior AI engineer requirements with strong Python and LLM product experience, proven cloud deployment on AWS, measurable impact in a logistics setting, and solid leadership and evaluation skills.

Hidden from the scorer: contact block, date of birth, email, gender, marital status, name, phone, photo, profile link

| Criterion | Score | Evidence | Note |
|---|---|---|---|
| skills_match | 5 | “Python, LangGraph, LangChain, FastAPI, AWS, Docker, PostgreSQL, pgvector, evaluation design”; “Built a RAG assistant over 40,000 shipment SOPs used by 1,200 dispatchers”; “Rebuilt the assistant as a LangGraph agent with tool calls into the order database”; “Created a 300-question golden set and an LLM-judge regression gate that blocks releases below 92% accuracy”; “Runs the service on AWS ECS with CloudWatch dashboards” |  |
| relevant_experience | 5 | “AI engineer with 7 years of Python experience, the last 3 building LLM products”; “Lead ML Engineer, ShipFast Logistics (2023 – present)”; “Software Engineer, Finlytics (2019 – 2023)” |  |
| delivery_impact | 5 | “Built a RAG assistant over 40,000 shipment SOPs used by 1,200 dispatchers; cut ticket handling time by 31%”; “reduced token cost by 45% with caching” |  |
| communication | 5 | “Mentored three junior engineers” |  |

Strengths: Extensive Python expertise (7 years); End‑to‑end LLM product development (RAG, LangGraph agents); AWS cloud deployment and operations; Design of evaluation pipelines (golden sets, LLM judges); Leadership and mentoring experience; Domain knowledge in logistics

Gaps: No explicit Kubernetes experience mentioned; Robotics, IoT domain experience not mentioned; Azure or GCP experience not mentioned

### Interview kit

1. **Can you walk us through the end‑to‑end architecture you built for a Retrieval‑Augmented Generation (RAG) system that answered operators' questions about their robot fleet?**  
   Probes: Depth of knowledge in RAG pipelines, vector store selection, prompt engineering, scaling, latency handling, and evaluation mechanisms.  
   Strong answer: A strong answer describes the full data flow: ingestion of robot telemetry, chunking and embedding (e.g., using OpenAI embeddings), storage in a vector database (e.g., Pinecone or FAISS), retrieval logic, prompt template that combines retrieved context with operator queries, and LLM inference. The candidate explains why they chose a particular vector store (e.g., latency, cost, scalability), how they handled large‑scale queries (batching, caching), measures taken to keep latency < 200 ms, and how they built an evaluation pipeline with golden sets and LLM judges to continuously monitor answer quality.
2. **Describe the process you used to take an LLM‑based feature from a prototype stage to a production service, focusing on monitoring, alerting, and cost control.**  
   Probes: Experience with productionizing LLM services, CI/CD pipelines, observability, cost‑optimization strategies, and reliability engineering.  
   Strong answer: The candidate outlines a CI/CD workflow (e.g., GitHub Actions deploying to AWS ECS/EKS), automated testing (unit, integration, prompt regression), and the observability stack (Prometheus metrics for request latency, token usage, error rates; Grafana dashboards; alerts via PagerDuty). They discuss cost controls such as token‑based budgeting, dynamic model selection (cheaper models for low‑risk queries), autoscaling policies, and periodic cost reviews. They also mention fallback mechanisms (e.g., cached responses) and post‑deployment validation steps.
3. **Our platform runs many services on Kubernetes. Although you haven’t listed Kubernetes experience, can you explain the core concepts you would need to master and how you would approach learning and applying them to this role?**  
   Probes: Awareness of Kubernetes fundamentals, ability to self‑learn, and plan for bridging the gap.  
   Strong answer: A strong answer acknowledges the gap, then demonstrates knowledge of key concepts: pods, deployments, services, ConfigMaps, Secrets, Horizontal Pod Autoscaling, and Helm charts. The candidate outlines a concrete learning plan (e.g., completing the CNCF Kubernetes Fundamentals course, hands‑on labs on EKS, building a small test deployment). They also discuss how their existing AWS experience (ECS/ECR, CloudWatch) provides a foundation and how they would leverage that to transition to Kubernetes, emphasizing a timeline and measurable milestones.
4. **The role requires deep understanding of robotics and IoT data streams. How would you leverage your logistics domain experience to quickly get up to speed with robot fleet telemetry and constraints?**  
   Probes: Transferable domain knowledge, learning strategy for robotics/IoT, and ability to apply existing expertise to new contexts.  
   Strong answer: The candidate draws parallels between logistics data (e.g., shipment tracking, route optimization) and robot telemetry (e.g., status, location, sensor streams). They describe a plan to study robot SDKs/APIs, attend internal domain workshops, and prototype a small data ingestion pipeline using sample robot data. They mention prior experience handling high‑velocity, noisy data and applying schema validation, which would translate to IoT streams. They also reference any past exposure to similar real‑time systems (e.g., fleet management) and how they would collaborate with domain experts.
5. **Design a scalable system that lets operators query the robot fleet using natural language, incorporating retrieval, LLM inference, and real‑time constraints. Include components for data ingestion, indexing, query handling, monitoring, and cost management.**  
   Probes: System design skills, ability to balance latency, scalability, security, and cost; understanding of relevant AWS services and LLM integration.  
   Strong answer: A strong answer presents a high‑level architecture: (1) Data ingestion layer using AWS IoT Core/Kinesis to collect telemetry, (2) Stream processing (AWS Lambda or Flink) to chunk and embed data, (3) Vector store (e.g., Amazon OpenSearch with k‑NN plugin or Pinecone) for retrieval, (4) API gateway exposing a query endpoint, (5) Retrieval service that fetches top‑k contexts, (6) Prompt orchestration service that builds the final prompt and calls the LLM (e.g., Amazon Bedrock or hosted OpenAI), (7) Response caching layer (Redis) for frequent queries, (8) Monitoring stack (CloudWatch metrics for latency, token usage, error rates) with alerts, (9) Cost controls via token budgeting, model selection, and autoscaling policies. The candidate discusses fault tolerance (retry logic, circuit breakers), security (IAM roles, VPC isolation), and how they would validate answer quality with automated golden‑set tests.
6. **You’ll be mentoring two mid‑level engineers. Can you share a specific example of how you helped an engineer grow technically and what measurable impact it had on the project?**  
   Probes: Mentorship approach, coaching techniques, outcomes, and ability to develop talent.  
   Strong answer: The candidate describes a concrete scenario: an engineer struggled with designing a robust evaluation pipeline. The senior engineer set up weekly 1‑on‑1s, introduced them to golden‑set creation, taught them how to write parameterized pytest suites, and paired on code reviews. Over three months, the engineer independently built an automated evaluation framework that reduced manual QA time by 40% and increased detection of regressions from 70% to 95%. The answer includes feedback loops, documentation they created, and how the mentorship aligned with team goals.

## C-02 — Li Wei

The candidate exceeds the required experience level with 8 years of Python and cloud work, has built a RAG chatbot prototype for internal users, and operates services on GCP. They meet all must‑haves, demonstrate strong technical and leadership skills, but lack several nice‑to‑have items such as LangGraph, formal LLM evaluation, and domain experience in robotics/logistics/IoT.

Hidden from the scorer: contact block, dob, email, gender, name, phone, religion

| Criterion | Score | Evidence | Note |
|---|---|---|---|
| skills_match | 4 | “Backend engineer with 8 years of Python and cloud experience, moving into AI.”; “Owns Python microservices on Google Cloud (GKE, Pub/Sub) serving 5 million requests a day.”; “Built an internal RAG chatbot prototype over engineering docs for 60 engineers; it is in pilot.”; “Python, Go, Kubernetes, GCP, Terraform, Redis, Elasticsearch, LangChain (prototype)” |  |
| relevant_experience | 5 | “Backend engineer with 8 years of Python and cloud experience, moving into AI.”; “Senior Backend Engineer, CloudCart (2020 – present)”; “Built an internal RAG chatbot prototype over engineering docs for 60 engineers; it is in pilot.” |  |
| delivery_impact | 4 | “Owns Python microservices on Google Cloud (GKE, Pub/Sub) serving 5 million requests a day.”; “Built an internal RAG chatbot prototype over engineering docs for 60 engineers; it is in pilot.” |  |
| communication | 4 | “Led the migration of 30 services to Kubernetes with zero downtime.”; “Senior Backend Engineer, CloudCart (2020 – present)” |  |

Strengths: 8 years of Python development; Extensive cloud experience on GCP; Kubernetes deployment and migration expertise; Built and piloted an internal RAG chatbot; Leadership in large‑scale service migration

Gaps: No explicit experience with LangGraph or similar agent orchestration framework; No mention of systematic LLM evaluation (golden sets, LLM judges, regression gates); No Kubernetes experience on AWS or Azure (only GCP); No robotics, logistics, or IoT domain experience; LLM application is a prototype/pilot, not a full production release to external users

### Interview kit

1. **Can you walk us through the architecture and productionization steps of the internal RAG chatbot you built, focusing on retrieval pipelines, LLM integration, and how you handled scaling and cost monitoring?**  
   Probes: Depth of experience with retrieval‑augmented generation, vector stores, prompt engineering, and moving a prototype to a production‑grade service.  
   Strong answer: A strong answer describes a clear end‑to‑end architecture (e.g., document ingestion → embedding generation → vector DB (e.g., Pinecone/FAISS) → retrieval service → LLM inference layer). It explains prompt design, handling latency (caching, batch inference), observability (metrics, tracing), cost controls (token usage limits, autoscaling policies), CI/CD pipelines, and any incident response processes. Mention of automated tests, monitoring dashboards, and concrete cost figures demonstrates production readiness.
2. **Describe the large‑scale service migration you led on GCP, including how you leveraged Kubernetes, CI/CD, and observability tools to ensure a smooth transition with minimal downtime.**  
   Probes: Experience with cloud migrations, Kubernetes deployment, and operational excellence on GCP.  
   Strong answer: A strong answer outlines a migration assessment (dependency mapping, risk analysis), refactoring for containerization, use of GKE with autoscaling, IaC (Terraform/Deployment Manager), CI/CD (Cloud Build, Argo CD) for zero‑downtime rollouts, and observability (Stackdriver, Prometheus, Grafana). It includes strategies like blue‑green or canary deployments, rollback plans, post‑migration validation, and quantifiable outcomes such as reduced latency or cost savings.
3. **You haven’t listed experience with LangGraph or similar agent orchestration frameworks. How would you approach learning and applying such a framework to build complex agent workflows for our robot fleet operators?**  
   Probes: Ability to quickly acquire new tooling, conceptual understanding of agent orchestration, and adaptability to framework‑specific patterns.  
   Strong answer: A strong answer acknowledges the gap, references familiarity with analogous tools (e.g., Airflow, Dagster, state machines), and outlines a learning plan: reviewing documentation, building a small proof‑of‑concept, mapping existing workflow patterns to LangGraph primitives, and integrating with existing services. It also discusses design considerations like modularity, error handling, and how to evaluate the framework’s fit.
4. **What processes would you put in place to create systematic LLM evaluation sets and quality gates, ensuring regression detection and consistent output quality over time?**  
   Probes: Knowledge of LLM evaluation methodologies, metric selection, and continuous quality assurance.  
   Strong answer: A strong answer describes building a golden dataset with representative queries, defining metrics (BLEU, ROUGE, factuality, latency, cost), setting up human‑in‑the‑loop evaluations, and automated regression tests that run on every model update. It includes CI integration, threshold alerts, versioned evaluation suites, and a feedback loop to retrain or fine‑tune models. Mention of tools like EvalAI, LLM judges, or custom judges shows practical experience.
5. **Design a system that provides retrieval‑augmented, agent‑driven answers to operators about their robot fleet, considering multi‑modal data sources, scalability, cost control, and monitoring. Outline the main components and data flow.**  
   Probes: System design skills specific to the role, including handling diverse data, cost‑aware LLM usage, and robust monitoring.  
   Strong answer: A strong answer presents a high‑level diagram: data ingestion layer (telemetry, logs, images) → preprocessing → embeddings stored in a scalable vector DB (e.g., Vertex AI Matching Engine). A retrieval service fetches relevant chunks, which feed into an orchestrated agent (using LangGraph or similar) that may call multiple LLMs or tools. The agent outputs are post‑processed, cached, and served via an API gateway with auth. Supporting services include a task queue (Pub/Sub), monitoring (Cloud Monitoring, OpenTelemetry), cost dashboards, and fallback mechanisms. The answer discusses scaling (horizontal pod autoscaling), latency budgets, cost caps (token limits), and observability (traces, alerts).
6. **How have you mentored mid‑level engineers in the past, and what specific practices do you use to help them grow technically and professionally?**  
   Probes: Mentorship approach, leadership style, and impact on team development.  
   Strong answer: A strong answer details regular one‑on‑ones, setting clear growth goals, pairing on complex tickets, code review best practices, and encouraging ownership of features. It mentions facilitating learning sessions, tracking progress, providing constructive feedback, and helping with career planning (e.g., preparing for promotions). Concrete outcomes such as improved code quality metrics, faster delivery times, or engineers taking on lead roles illustrate effectiveness.

## C-04 — Sara Khan

The candidate shows solid Python and Azure RAG skills with a shipped product for 300 customers, but lacks the required years of experience and several nice‑to‑have technologies.

Hidden from the scorer: age, contact block, email, gender, name, phone

| Criterion | Score | Evidence | Note |
|---|---|---|---|
| skills_match | 3 | “Python”; “Azure OpenAI”; “prompt evaluation” |  |
| relevant_experience | 1 | “Junior AI engineer with 2 years of experience.” |  |
| delivery_impact | 4 | “Shipped a customer-support RAG bot to 300 business customers on Azure.” |  |
| communication | 3 | “The candidate ships quickly and writes clearly.”; “Wrote the evaluation notebook used to compare prompt versions.” |  |

Strengths: Python programming; RAG application development; Azure cloud deployment; Prompt evaluation

Gaps: Insufficient total experience (needs 5+ years, has 2); Python production experience < 3 years; No LangGraph or other agent orchestration framework mentioned; No Kubernetes experience; No robotics, logistics or IoT domain experience

## C-05 — Tomás Herrera

The candidate has solid research experience and strong Python/ML skills but lacks the production‑grade engineering, cloud deployment, and shipped LLM product experience required for a Senior AI Engineer role.

Hidden from the scorer: age, contact block, email, gender, name, nationality, phone

| Criterion | Score | Evidence | Note |
|---|---|---|---|
| skills_match | 2 | “Python, PyTorch, Hugging Face Transformers, LaTeX, retrieval evaluation (benchmarks)”; “Built research prototypes of retrieval systems evaluated on public benchmarks.” |  |
| relevant_experience | 3 | “Machine learning researcher with 6 years of experience in deep learning.”; “Research Scientist, Universidad Politécnica AI Lab (2020 – present)” |  |
| delivery_impact | 0 | — |  |
| communication | 2 | “Published 5 papers on retrieval-augmented language models at ACL and EMNLP.”; “Machine learning researcher with 6 years of experience in deep learning.” |  |

Strengths: 6 years of deep learning research experience; Publications at top conferences (ACL, EMNLP); Experience with retrieval‑augmented language models; Proficiency in Python and PyTorch

Gaps: No evidence of Python used in production environments; No shipped LLM product to real users; No cloud deployment (AWS/GCP/Azure) experience; Missing agent orchestration frameworks (e.g., LangGraph); No Kubernetes or container orchestration experience; No robotics, logistics, or IoT domain experience

## C-03 — Omar Haddad

The candidate has a solid decade of Java and Azure cloud experience and has led engineering teams, but lacks the Python, LLM, and AI‑specific experience required for a Senior AI Engineer role.

Hidden from the scorer: contact block, email, marital status, name, nationality, phone

| Criterion | Score | Evidence | Note |
|---|---|---|---|
| skills_match | 1 | “Java, Spring Boot, Azure, Oracle, Kafka.”; “Leads a team of 8 building Spring Boot services on Azure.” |  |
| relevant_experience | 1 | “Enterprise Java developer with 10 years of experience.” |  |
| delivery_impact | 1 | “Leads a team of 8 building Spring Boot services on Azure.” |  |
| communication | 2 | “Leads a team of 8 building Spring Boot services on Azure.”; “Introduced CI/CD with Azure DevOps.” |  |

Strengths: 10 years of enterprise software development; Strong Java and Spring Boot expertise; Experience leading a team of 8 engineers; Hands‑on experience with Azure cloud services and CI/CD

Gaps: No Python production experience; No evidence of building or shipping LLM applications; No experience with LangGraph, agent orchestration, or LLM evaluation frameworks; No Kubernetes, robotics, logistics, or IoT domain experience

## Job description (approved)

# About us
Northwind Robotics builds warehouse robots and the FleetOS software that runs them.

# The role
We are looking for a Senior AI Engineer to join our team. The position can be based in Pune, India (hybrid) or remote anywhere within India. The role involves developing and productionising large‑language‑model (LLM) features that help operators interact with robot fleets.

# What you will do
- Build retrieval and agent features that answer operators' questions about their robot fleet.
- Take LLM prototypes to production, including monitoring, cost control and reliability.
- Design evaluation sets and quality gates for LLM outputs.
- Mentor two mid‑level engineers.

# Must have
- Python in production for at least 3 years.
- Experience building and shipping an LLM application (RAG, agents or similar) to real users.
- Experience deploying and operating services on a public cloud (AWS, GCP or Azure).

# Nice to have
- Experience with LangGraph or another agent‑orchestration framework.
- Experience evaluating LLM systems (golden sets, LLM judges, regression gates).
- Knowledge of Kubernetes.
- Background in robotics, logistics or IoT domains.

# What we offer
- Salary range: INR 45 – 60 lakh per year.
- Health cover for family.
- Learning budget of INR 1.5 lakh per year.
- Four weeks of paid leave.
- Flexible hybrid or remote work within India.

# Location and pay
- Location: Pune, India (hybrid) or remote within India.
- Compensation: INR 45 – 60 lakh per year.
- Interview window: 2026‑11‑02 to 2026‑11‑13.