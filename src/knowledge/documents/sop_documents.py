# File: src/knowledge/documents/sop_documents.py
"""
Standard Operating Procedures (SOPs) for Clinical Trial Operations
Comprehensive SOPs for data management, monitoring, safety, and site management.
"""

SOP_DOCUMENTS = {
    "document_info": {
        "title": "Clinical Trial Standard Operating Procedures",
        "version": "1.0.0",
        "last_updated": "2024-12-31",
        "department": "Clinical Operations",
        "purpose": "Provide standardized procedures for clinical trial operations"
    },
    
    "sops": [
        # SOP 1: Data Management
        {
            "sop_id": "SOP-DM-001",
            "title": "Data Entry and Verification Procedures",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Data Management",
            "category": "data_management",
            "objective": "To establish procedures for accurate and timely data entry and verification in clinical trials.",
            "scope": "This SOP applies to all clinical trial data entry personnel, data managers, and CRAs involved in data collection and verification.",
            "content": """
1. PURPOSE
This SOP describes the procedures for entering clinical trial data into the Electronic Data Capture (EDC) system and verifying data accuracy.

2. RESPONSIBILITIES

2.1 Site Coordinator:
- Enter data within 3 business days of subject visit
- Respond to queries within 5 business days
- Verify data accuracy against source documents
- Obtain required signatures

2.2 Data Manager:
- Review data for completeness and consistency
- Generate and manage data queries
- Perform data validation checks
- Track data entry metrics

2.3 CRA:
- Perform source data verification
- Identify data discrepancies
- Document findings in monitoring reports
- Follow up on outstanding issues

3. PROCEDURES

3.1 Data Entry
a) Log into EDC system with unique credentials
b) Select appropriate subject and visit
c) Enter all required data fields
d) Review entries before saving
e) Submit for query/review as applicable

3.2 Data Verification
a) Compare EDC data to source documents
b) Verify dates, values, and units
c) Check for transcription errors
d) Document any discrepancies
e) Generate queries for corrections

3.3 Query Management
a) Review open queries daily
b) Investigate data discrepancy
c) Provide complete response with source reference
d) Make corrections as appropriate
e) Verify query closure

3.4 Timeline Requirements
- Initial data entry: Within 3 business days of visit
- Query response: Within 5 business days
- SDV completion: Per monitoring plan
- Signature completion: Within 7 business days

4. QUALITY CONTROL
- Weekly data entry compliance review
- Monthly query aging analysis
- Quarterly process review
- Annual SOP review

5. DOCUMENTATION
- All entries logged in EDC audit trail
- Query responses documented
- SDV documented in monitoring reports
- Training records maintained

6. REFERENCES
- ICH E6(R2) Section 4.9
- 21 CFR Part 11
- Company Data Management Plan
            """,
            "keywords": ["data entry", "verification", "EDC", "queries", "source documents"],
            "roles": ["Site Coordinator", "Data Manager", "CRA"],
            "related_sops": ["SOP-DM-002", "SOP-DM-003", "SOP-MO-001"]
        },
        
        {
            "sop_id": "SOP-DM-002",
            "title": "Query Generation and Resolution",
            "version": "2.5",
            "effective_date": "2024-01-01",
            "department": "Data Management",
            "category": "data_management",
            "objective": "To establish standardized procedures for generating, tracking, and resolving data queries.",
            "scope": "This SOP applies to data managers, site coordinators, and CRAs involved in query management.",
            "content": """
1. PURPOSE
This SOP defines the procedures for managing data queries throughout the clinical trial lifecycle.

2. QUERY TYPES

2.1 Automatic Queries
- System-generated based on edit checks
- Range checks, cross-field validations
- Required field completeness

2.2 Manual Queries
- Data manager generated
- CRA generated during monitoring
- Medical monitor queries

2.3 Query Categories
- DM Query: Data management issues
- Clinical Query: Clinical/medical questions
- Safety Query: Safety-related clarifications
- Coding Query: Coding clarification needed

3. PROCEDURES

3.1 Query Generation
a) Identify data discrepancy or question
b) Verify discrepancy is valid
c) Write clear, specific query text
d) Select appropriate query type
e) Assign to appropriate recipient
f) Set priority level

3.2 Query Text Standards
- Be specific about the issue
- Reference the data in question
- Ask a clear question
- Avoid leading the response
- Include source reference if applicable

EXAMPLE:
"The reported lab value of 150 mg/dL for glucose on 15-Jan-2024 appears to be outside the expected range. Please verify this value against source documents and confirm or correct."

3.3 Query Response
a) Review query details
b) Check source documents
c) Provide complete response
d) Include source document reference
e) Make corrections if needed
f) Do not close query without resolution

3.4 Query Escalation
- Day 5: First reminder (auto-generated)
- Day 10: CRA follow-up
- Day 15: CTM notification
- Day 21: Study Lead escalation
- Day 30: Sponsor notification

4. QUERY AGING CATEGORIES
- Current (0-5 days): Normal processing
- Aging (6-10 days): Attention needed
- Overdue (11-20 days): Escalation triggered
- Critical (>20 days): Immediate action required

5. QUERY METRICS
- Query rate per subject
- Query resolution time
- Query re-open rate
- Queries by type/category
- Aging distribution

6. QUALITY CONTROL
- Daily review of new queries
- Weekly aging review
- Monthly trend analysis
- Quarterly process optimization
            """,
            "keywords": ["query", "resolution", "escalation", "data clarification", "aging"],
            "roles": ["Data Manager", "Site Coordinator", "CRA"],
            "related_sops": ["SOP-DM-001", "SOP-DM-003", "SOP-MO-002"]
        },
        
        {
            "sop_id": "SOP-DM-003",
            "title": "Database Lock Procedures",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Data Management",
            "category": "data_management",
            "objective": "To establish procedures for preparing and executing clinical trial database lock.",
            "scope": "This SOP applies to data management, clinical operations, and biostatistics personnel involved in database lock activities.",
            "content": """
1. PURPOSE
This SOP describes the procedures for achieving database lock in clinical trials, ensuring data integrity and regulatory compliance.

2. PRE-LOCK REQUIREMENTS

2.1 Data Completeness
- All CRF pages entered
- All required visits completed
- All protocol deviations documented
- All AEs/SAEs entered

2.2 Data Quality
- All queries resolved (0 open queries)
- All edit checks passed
- All required SDV complete
- All signatures obtained

2.3 Coding Complete
- All AEs coded (MedDRA)
- All medications coded (WHODrug)
- Coding review complete
- Dictionary versions documented

2.4 External Data
- Lab data reconciled
- ECG data reconciled
- Central assessments complete
- EDRR issues resolved

3. LOCK PROCEDURES

3.1 Pre-Lock Checklist
☐ Data entry complete
☐ All queries closed
☐ SDV 100% complete
☐ All signatures obtained
☐ Coding complete
☐ SAE reconciliation complete
☐ External data reconciled
☐ Medical review complete

3.2 Lock Execution
a) Verify pre-lock checklist complete
b) Obtain lock approvals
c) Execute soft lock (freeze)
d) Perform final data review
e) Execute hard lock
f) Generate lock documentation
g) Archive database

3.3 Lock Approvals Required
- Data Management Lead
- Medical Monitor
- Biostatistics Lead
- Study Manager
- Sponsor Representative

4. POST-LOCK PROCEDURES

4.1 Documentation
- Lock completion memo
- Final data transfer
- Archive procedures
- Audit trail export

4.2 Unlock Procedures (if needed)
- Formal unlock request required
- Reason documented
- Limited scope unlock preferred
- Re-lock with documentation

5. TIMELINE
- 4 weeks pre-lock: 80% subjects clean
- 2 weeks pre-lock: 95% subjects clean
- 1 week pre-lock: All subjects frozen
- Lock day: Final lock execution
- Post-lock: Analysis datasets created

6. QUALITY CONTROL
- Daily pre-lock status review
- Lock readiness meetings
- Post-lock verification
- Documentation audit
            """,
            "keywords": ["database lock", "freeze", "data integrity", "pre-lock", "post-lock"],
            "roles": ["Data Manager", "Study Manager", "Biostatistician"],
            "related_sops": ["SOP-DM-001", "SOP-DM-002", "SOP-ST-001"]
        },
        
        # SOP 2: Monitoring
        {
            "sop_id": "SOP-MO-001",
            "title": "Site Monitoring Procedures",
            "version": "4.0",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "monitoring",
            "objective": "To establish procedures for conducting clinical trial monitoring visits.",
            "scope": "This SOP applies to CRAs, CTMs, and clinical operations personnel involved in site monitoring.",
            "content": """
1. PURPOSE
This SOP describes the procedures for planning, conducting, and documenting clinical trial monitoring visits.

2. MONITORING APPROACH

2.1 Risk-Based Monitoring
- Centralized monitoring as primary method
- On-site monitoring based on risk assessment
- Targeted SDV for critical data
- Key Risk Indicators (KRIs) tracking

2.2 Visit Types
- Site Initiation Visit (SIV)
- Interim Monitoring Visit (IMV)
- Close-out Visit (COV)
- For-cause Visit

3. PRE-VISIT PROCEDURES

3.1 Planning
a) Review previous visit report
b) Review centralized monitoring data
c) Identify focus areas
d) Schedule visit with site
e) Prepare visit agenda
f) Review outstanding action items

3.2 Pre-Visit Letter
- Confirm date and time
- List documents needed
- Identify subjects for review
- Request staff availability
- Send 2 weeks in advance

4. ON-SITE PROCEDURES

4.1 Opening Meeting
- Review visit objectives
- Confirm agenda
- Identify available staff
- Review study status

4.2 Document Review
- Regulatory documents
- Delegation log
- Training records
- Essential documents

4.3 Subject Review
- Informed consent
- Eligibility verification
- Source data verification
- AE/SAE documentation
- Protocol compliance

4.4 Investigational Product
- Accountability review
- Storage conditions
- Dispensing records
- Returns/destruction

4.5 Closing Meeting
- Summarize findings
- Discuss action items
- Set timelines
- Thank site staff

5. POST-VISIT PROCEDURES

5.1 Monitoring Report
- Complete within 10 business days
- Document all findings
- List action items with due dates
- Distribute per SOP

5.2 Follow-up
- Track action items
- Provide support as needed
- Escalate if issues persist

6. SDV REQUIREMENTS
- 100% for critical data points
- Per monitoring plan for other data
- Document in monitoring report
- Generate queries for discrepancies
            """,
            "keywords": ["monitoring", "site visit", "CRA", "SDV", "risk-based monitoring"],
            "roles": ["CRA", "CTM", "Study Manager"],
            "related_sops": ["SOP-MO-002", "SOP-MO-003", "SOP-SI-001"]
        },
        
        {
            "sop_id": "SOP-MO-002",
            "title": "Monitoring Report Completion",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "monitoring",
            "objective": "To establish procedures for completing and distributing monitoring visit reports.",
            "scope": "This SOP applies to CRAs and CTMs responsible for monitoring report documentation.",
            "content": """
1. PURPOSE
This SOP describes the requirements for completing monitoring visit reports in a timely and comprehensive manner.

2. REPORT TIMING
- Draft completion: Within 5 business days
- Final submission: Within 10 business days
- Critical findings: Reported within 24 hours

3. REPORT SECTIONS

3.1 Administrative Information
- Visit date and type
- Site identification
- CRA name and signature
- Staff present
- Visit duration

3.2 Subject Status Summary
- Enrolled subjects
- Screen failures
- Discontinued subjects
- Completed subjects
- Active subjects

3.3 Informed Consent Review
- Number reviewed
- Issues identified
- Re-consent requirements
- Resolution status

3.4 Source Data Verification
- Subjects reviewed
- Data points verified
- Discrepancies found
- Queries generated

3.5 Investigational Product Review
- Accountability verified
- Storage conditions checked
- Expiry dates reviewed
- Discrepancies noted

3.6 Essential Documents
- Documents reviewed
- Updates required
- Filing compliance
- Missing documents

3.7 Protocol Deviations
- New deviations identified
- Previous deviation status
- Corrective actions

3.8 Adverse Event Review
- AEs reviewed
- SAEs verified
- Reporting compliance
- Follow-up status

3.9 Action Items
- Previous items status
- New items identified
- Responsible party
- Due dates

4. REPORT QUALITY
- Clear, factual language
- No opinions without evidence
- Consistent with previous reports
- Proofread before submission

5. REPORT DISTRIBUTION
- Sponsor/CRO
- TMF
- Site file (if required)
- Regulatory file

6. ARCHIVING
- Part of Trial Master File
- Retained per retention requirements
- Available for audit/inspection
            """,
            "keywords": ["monitoring report", "documentation", "visit report", "findings", "action items"],
            "roles": ["CRA", "CTM"],
            "related_sops": ["SOP-MO-001", "SOP-MO-003", "SOP-ES-001"]
        },
        
        {
            "sop_id": "SOP-MO-003",
            "title": "Centralized Monitoring Procedures",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "monitoring",
            "objective": "To establish procedures for conducting centralized monitoring activities.",
            "scope": "This SOP applies to data managers, CRAs, and clinical operations personnel involved in centralized monitoring.",
            "content": """
1. PURPOSE
This SOP describes the procedures for conducting centralized monitoring as part of a risk-based monitoring approach.

2. CENTRALIZED MONITORING COMPONENTS

2.1 Data Analytics
- Statistical analyses of trial data
- Cross-site comparisons
- Trend identification
- Outlier detection

2.2 Key Risk Indicators (KRIs)
- Enrollment metrics
- Data quality metrics
- Protocol compliance metrics
- Safety metrics

2.3 Remote Data Review
- EDC data review
- Query status review
- Data completeness assessment
- Timeliness tracking

3. KRI DEFINITIONS

3.1 Enrollment KRIs
- Enrollment rate vs. target
- Screen failure rate
- Randomization rate
- Discontinuation rate

3.2 Data Quality KRIs
- Data entry timeliness
- Query rate per subject
- Query aging
- Missing data rate
- SDV completion rate

3.3 Protocol Compliance KRIs
- Protocol deviation rate
- Visit window compliance
- Procedure compliance

3.4 Safety KRIs
- AE reporting rate
- SAE reporting compliance
- Safety signal detection

4. THRESHOLDS AND TRIGGERS

4.1 Green (Normal)
- Within expected range
- No action required
- Continue routine monitoring

4.2 Yellow (Warning)
- Approaching threshold
- Enhanced monitoring
- Root cause investigation

4.3 Red (Action Required)
- Threshold exceeded
- Immediate investigation
- Potential for-cause visit
- Escalation to management

5. REPORTING

5.1 KRI Dashboard
- Updated weekly/monthly
- Visual trend display
- Site comparison
- Drill-down capability

5.2 Centralized Monitoring Report
- Monthly summary
- Site-specific findings
- Recommended actions
- Trend analysis

6. INTEGRATION WITH ON-SITE MONITORING
- Inform visit planning
- Focus areas for SDV
- Guide resource allocation
- Support risk assessment
            """,
            "keywords": ["centralized monitoring", "KRI", "risk indicators", "remote monitoring", "analytics"],
            "roles": ["Data Manager", "CRA", "CTM"],
            "related_sops": ["SOP-MO-001", "SOP-MO-002", "SOP-DM-001"]
        },
        
        # SOP 3: Safety
        {
            "sop_id": "SOP-SA-001",
            "title": "Serious Adverse Event Reporting",
            "version": "5.0",
            "effective_date": "2024-01-01",
            "department": "Drug Safety",
            "category": "safety",
            "objective": "To establish procedures for reporting serious adverse events in clinical trials.",
            "scope": "This SOP applies to all personnel involved in SAE identification, documentation, and reporting.",
            "content": """
1. PURPOSE
This SOP describes the procedures for identifying, documenting, and reporting serious adverse events (SAEs) in clinical trials.

2. SAE DEFINITION
A Serious Adverse Event is any untoward medical occurrence that:
- Results in death
- Is life-threatening
- Requires inpatient hospitalization or prolongs existing hospitalization
- Results in persistent or significant disability/incapacity
- Is a congenital anomaly/birth defect
- Is an important medical event

3. REPORTING TIMELINES

3.1 Initial Report
- Fatal/Life-threatening: Within 24 hours
- Other SAEs: Within 24-72 hours (per protocol)

3.2 Follow-up Reports
- Within 24 hours of new information
- Continue until resolution

4. PROCEDURES

4.1 SAE Identification
a) Investigator identifies SAE
b) Assess seriousness criteria
c) Determine relationship to study drug
d) Document in source records

4.2 Initial Reporting
a) Complete SAE form
b) Include all available information
c) Submit to sponsor within timeline
d) Retain copy in site files

4.3 SAE Form Completion
Required information:
- Subject identifier
- Event description
- Onset and resolution dates
- Seriousness criteria
- Severity (mild/moderate/severe)
- Relationship to study drug
- Action taken with study drug
- Outcome
- Reporter information

4.4 Follow-up Reporting
a) Gather additional information
b) Update SAE form
c) Document resolution
d) Submit follow-up report

5. REGULATORY REPORTING

5.1 Expedited Reporting
- SUSARs to regulatory authorities
- Per regional requirements
- Sponsor responsibility

5.2 IRB/IEC Notification
- Per IRB requirements
- Annual safety reports
- Significant safety issues

6. RECONCILIATION
- Compare DM and Safety databases
- Identify discrepancies
- Resolve within 5 business days
- Document reconciliation

7. QUALITY CONTROL
- Daily review of new SAEs
- Weekly reconciliation check
- Monthly compliance review
- Quarterly audit
            """,
            "keywords": ["SAE", "serious adverse event", "safety reporting", "pharmacovigilance", "expedited"],
            "roles": ["Investigator", "Site Coordinator", "Safety Officer", "Medical Monitor"],
            "related_sops": ["SOP-SA-002", "SOP-SA-003", "SOP-DM-001"]
        },
        
        {
            "sop_id": "SOP-SA-002",
            "title": "SAE Reconciliation Procedures",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Drug Safety",
            "category": "safety",
            "objective": "To establish procedures for reconciling SAE data between clinical and safety databases.",
            "scope": "This SOP applies to data management and drug safety personnel involved in SAE reconciliation.",
            "content": """
1. PURPOSE
This SOP describes the procedures for ensuring alignment of SAE data between data management (EDC) and drug safety databases.

2. RECONCILIATION FREQUENCY
- Ongoing: As SAEs are received
- Formal: Monthly reconciliation
- Pre-lock: Final reconciliation

3. RECONCILIATION SCOPE

3.1 Data Points to Reconcile
- SAE presence (listed in both systems)
- Subject identifier
- Event term/description
- Onset date
- Resolution date
- Outcome
- Causality assessment
- Seriousness criteria

3.2 Acceptable Differences
- Minor wording variations
- Different coding levels
- Timing of updates

3.3 Discrepancies Requiring Resolution
- Missing SAE in one system
- Different onset/resolution dates
- Different causality assessment
- Different outcome

4. PROCEDURES

4.1 Data Extraction
a) Export SAE listing from EDC
b) Export SAE listing from safety database
c) Match on subject ID and event
d) Identify discrepancies

4.2 Discrepancy Investigation
a) Determine source of discrepancy
b) Review original documents
c) Contact site if needed
d) Determine correct data

4.3 Resolution
a) Correct errors in appropriate system
b) Document correction
c) Verify alignment
d) Close reconciliation item

4.4 Documentation
- Reconciliation log maintained
- Discrepancies documented
- Resolutions recorded
- Sign-off obtained

5. ESCALATION
- Day 3: DM/Safety follow-up
- Day 7: Manager notification
- Day 14: Study lead escalation

6. METRICS
- Reconciliation rate
- Time to resolution
- Discrepancy types
- Repeat discrepancies

7. PRE-LOCK REQUIREMENTS
- 100% reconciliation complete
- All discrepancies resolved
- Sign-off from DM and Safety
- Documentation archived
            """,
            "keywords": ["SAE reconciliation", "safety database", "discrepancy", "alignment", "data management"],
            "roles": ["Data Manager", "Safety Data Manager", "Medical Monitor"],
            "related_sops": ["SOP-SA-001", "SOP-SA-003", "SOP-DM-003"]
        },
        
        {
            "sop_id": "SOP-SA-003",
            "title": "Safety Signal Detection and Response",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Drug Safety",
            "category": "safety",
            "objective": "To establish procedures for detecting and responding to safety signals in clinical trials.",
            "scope": "This SOP applies to drug safety, medical monitoring, and clinical operations personnel.",
            "content": """
1. PURPOSE
This SOP describes the procedures for identifying, evaluating, and responding to safety signals during clinical trial conduct.

2. SIGNAL DEFINITION
A safety signal is information about a new or known adverse event that may require further investigation or action.

3. SIGNAL SOURCES
- Individual case reports
- Aggregate data analysis
- Literature reports
- Regulatory notifications
- Investigator reports
- DSMB recommendations

4. DETECTION METHODS

4.1 Quantitative Methods
- Disproportionality analysis
- Observed vs. expected comparisons
- Time-to-onset analysis
- Dose-response evaluation

4.2 Qualitative Methods
- Clinical case review
- Medical expert assessment
- Literature evaluation
- Regulatory intelligence

5. SIGNAL EVALUATION

5.1 Initial Assessment
- Validate signal
- Characterize event
- Assess causality
- Evaluate clinical significance

5.2 Signal Criteria
- Statistical threshold met
- Clinical plausibility
- Consistency across sources
- Temporal relationship

5.3 Documentation
- Signal detection form
- Evaluation summary
- Recommendation
- Decision rationale

6. SIGNAL RESPONSE

6.1 Actions May Include
- Enhanced monitoring
- Protocol amendment
- Informed consent update
- Investigator notification
- Regulatory notification
- Study suspension/termination

6.2 Communication
- Internal stakeholders
- Investigators
- IRBs/IECs
- Regulatory authorities
- DSMB

7. ONGOING SURVEILLANCE
- Continue monitoring after initial response
- Track signal over time
- Update assessment as needed
- Document resolution

8. DOCUMENTATION
- Signal detection log
- Evaluation reports
- Action documentation
- Communication records
            """,
            "keywords": ["safety signal", "detection", "surveillance", "pharmacovigilance", "DSMB"],
            "roles": ["Safety Officer", "Medical Monitor", "Study Manager"],
            "related_sops": ["SOP-SA-001", "SOP-SA-002", "SOP-MO-003"]
        },
        
        # SOP 4: Site Management
        {
            "sop_id": "SOP-SI-001",
            "title": "Site Selection and Initiation",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "site_management",
            "objective": "To establish procedures for selecting and initiating clinical trial sites.",
            "scope": "This SOP applies to clinical operations personnel involved in site selection and initiation.",
            "content": """
1. PURPOSE
This SOP describes the procedures for selecting appropriate sites and initiating them for clinical trial participation.

2. SITE SELECTION

2.1 Selection Criteria
- Investigator qualifications
- Patient population access
- Facility capabilities
- Staff experience
- Regulatory compliance history
- Previous performance (if applicable)

2.2 Feasibility Assessment
a) Questionnaire completion
b) Patient database review
c) Competitive trial assessment
d) Resource evaluation
e) Site visit (if needed)

2.3 Selection Decision
- Feasibility review
- Risk assessment
- Selection committee approval
- Documentation

3. PRE-INITIATION ACTIVITIES

3.1 Regulatory Requirements
- IRB/IEC submission
- Regulatory authority notification
- Contract negotiation
- Budget finalization

3.2 Essential Documents
- Collect required documents
- Review for completeness
- File in TMF
- Track outstanding items

3.3 Site Preparation
- EDC access setup
- IVRS/IWRS setup
- IP shipment planning
- Training scheduling

4. SITE INITIATION VISIT (SIV)

4.1 Pre-SIV
- Confirm all approvals received
- Schedule visit
- Prepare materials
- Send agenda

4.2 SIV Agenda
- Protocol review
- Procedure training
- EDC training
- IP handling
- Safety reporting
- Essential documents
- Q&A

4.3 Post-SIV
- Complete SIV report
- Document training
- Confirm site activation
- Update tracking systems

5. SITE ACTIVATION
- All essential documents complete
- Training documented
- IP received
- EDC access verified
- Activation letter issued

6. DOCUMENTATION
- Feasibility records
- Selection decision
- SIV report
- Activation confirmation
            """,
            "keywords": ["site selection", "initiation", "SIV", "feasibility", "activation"],
            "roles": ["CTM", "CRA", "Study Manager"],
            "related_sops": ["SOP-SI-002", "SOP-SI-003", "SOP-MO-001"]
        },
        
        {
            "sop_id": "SOP-SI-002",
            "title": "Site Performance Management",
            "version": "2.5",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "site_management",
            "objective": "To establish procedures for monitoring and managing site performance during clinical trials.",
            "scope": "This SOP applies to CRAs, CTMs, and clinical operations personnel responsible for site oversight.",
            "content": """
1. PURPOSE
This SOP describes the procedures for assessing, monitoring, and improving site performance throughout the clinical trial.

2. PERFORMANCE METRICS

2.1 Enrollment Metrics
- Enrollment rate
- Screen failure rate
- Randomization rate
- Dropout rate

2.2 Data Quality Metrics
- Data Quality Index (DQI)
- Query rate
- Query resolution time
- Missing data rate
- SDV completion

2.3 Compliance Metrics
- Protocol deviation rate
- Visit window compliance
- SAE reporting compliance
- Essential document compliance

2.4 Operational Metrics
- CRF completion time
- Query response time
- Signature turnaround
- Training compliance

3. PERFORMANCE TIERS

3.1 Exceptional (Top 10%)
- Benchmark performance
- Reduced monitoring
- Recognition consideration

3.2 Strong (70-90th percentile)
- Meeting expectations
- Standard monitoring
- Minor improvements

3.3 Average (30-70th percentile)
- Acceptable performance
- Standard monitoring
- Some attention needed

3.4 Below Average (10-30th percentile)
- Needs improvement
- Enhanced monitoring
- Action plan required

3.5 Needs Improvement (Bottom 10%)
- Significant concerns
- Intensive monitoring
- Remediation required
- Potential closure

4. PERFORMANCE IMPROVEMENT

4.1 Root Cause Analysis
- Identify performance gaps
- Determine underlying causes
- Consider site factors
- Evaluate support needs

4.2 Action Planning
- Define improvement goals
- Specify actions
- Assign responsibilities
- Set timelines

4.3 Implementation
- Execute action plan
- Provide support
- Monitor progress
- Adjust as needed

4.4 Evaluation
- Assess improvement
- Document outcomes
- Update site risk level
- Close or continue

5. ESCALATION

5.1 Triggers
- DQI < 70 for 2 months
- Query rate > 2x average
- > 5 major deviations
- SAE reporting failure

5.2 Process
- CTM notification
- Study lead involvement
- Sponsor notification (if needed)
- Site closure evaluation

6. DOCUMENTATION
- Performance reports
- Action plans
- Meeting minutes
- Improvement tracking
            """,
            "keywords": ["site performance", "metrics", "improvement", "monitoring", "escalation"],
            "roles": ["CRA", "CTM", "Study Manager"],
            "related_sops": ["SOP-SI-001", "SOP-SI-003", "SOP-MO-001"]
        },
        
        {
            "sop_id": "SOP-SI-003",
            "title": "Site Issue Escalation and Resolution",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Clinical Operations",
            "category": "site_management",
            "objective": "To establish procedures for escalating and resolving site issues during clinical trials.",
            "scope": "This SOP applies to all personnel involved in site management and issue resolution.",
            "content": """
1. PURPOSE
This SOP describes the procedures for identifying, escalating, and resolving site issues in clinical trials.

2. ISSUE CATEGORIES

2.1 Data Quality Issues
- High query rate
- Missing data
- Data entry delays
- SDV findings

2.2 Protocol Compliance Issues
- Protocol deviations
- Visit window violations
- Procedure non-compliance
- IC issues

2.3 Safety Issues
- SAE reporting delays
- Safety signal concerns
- AE documentation gaps
- Subject safety concerns

2.4 Operational Issues
- Staff turnover
- Resource constraints
- Communication problems
- Training gaps

3. ESCALATION LEVELS

3.1 Level 1: CRA Management (Days 1-7)
- Minor issues
- Routine follow-up
- Training/support provided
- Documented in visit report

3.2 Level 2: CTM Involvement (Days 8-14)
- Persistent issues
- Formal communication
- Action plan development
- Enhanced monitoring

3.3 Level 3: Study Lead (Days 15-21)
- Unresolved Level 2 issues
- Major compliance concerns
- Call with PI
- Written improvement plan

3.4 Level 4: Sponsor Notification (Days 22-30)
- Critical issues
- Subject safety concerns
- Formal warning
- Remediation required

3.5 Level 5: Closure Consideration (Day 30+)
- Failed remediation
- Continued non-compliance
- Closure evaluation
- Subject transfer planning

4. RESOLUTION PROCEDURES

4.1 Issue Documentation
- Describe issue clearly
- Document timeline
- List actions taken
- Track site response

4.2 Root Cause Analysis
- Identify underlying cause
- Consider contributing factors
- Avoid blame
- Focus on solutions

4.3 Action Plan
- Define corrective actions
- Assign responsibilities
- Set timelines
- Agree with site

4.4 Follow-up
- Monitor implementation
- Verify effectiveness
- Document resolution
- Close issue

5. SITE CLOSURE PROCEDURES

5.1 Closure Decision
- Document rationale
- Obtain approvals
- Plan subject transition
- Regulatory notifications

5.2 Closure Activities
- Transfer subjects (if applicable)
- Collect essential documents
- IP return
- Close-out visit
- Final documentation

6. DOCUMENTATION
- Issue log
- Escalation communications
- Action plans
- Resolution documentation
            """,
            "keywords": ["escalation", "site issues", "resolution", "compliance", "remediation"],
            "roles": ["CRA", "CTM", "Study Manager"],
            "related_sops": ["SOP-SI-001", "SOP-SI-002", "SOP-MO-001"]
        },
        
        # SOP 5: Essential Documents
        {
            "sop_id": "SOP-ES-001",
            "title": "Trial Master File Management",
            "version": "3.5",
            "effective_date": "2024-01-01",
            "department": "Quality Assurance",
            "category": "essential_documents",
            "objective": "To establish procedures for managing the Trial Master File throughout the clinical trial.",
            "scope": "This SOP applies to all personnel responsible for TMF documentation and maintenance.",
            "content": """
1. PURPOSE
This SOP describes the procedures for creating, maintaining, and archiving the Trial Master File (TMF).

2. TMF STRUCTURE

2.1 ICH E6(R2) Reference Model
- Zone 01: Trial Management
- Zone 02: Central Trial Documents
- Zone 03: Regulatory
- Zone 04: IRB/IEC
- Zone 05: Site Management
- Zone 06: IP
- Zone 07: Safety Reporting
- Zone 08: Statistics
- Zone 09: Data Management

2.2 Document Types
- Sponsor documents
- Site documents
- Country-specific documents
- Third-party documents

3. TMF QUALITY STANDARDS

3.1 Document Requirements (ALCOA+)
- Attributable: Author identifiable
- Legible: Readable and clear
- Contemporaneous: Created at time of event
- Original: Certified copies acceptable
- Accurate: Reflects actual events
- Complete: All required documents present
- Consistent: No contradictions
- Enduring: Retained for required period
- Available: Accessible when needed

3.2 Filing Requirements
- Correct location per taxonomy
- Proper naming convention
- Complete metadata
- Version controlled
- Signed/dated as required

4. TMF MAINTENANCE

4.1 Ongoing Activities
- Regular filing of documents
- Quality checks
- Completeness reviews
- Index updates

4.2 Inspection Readiness
- Document inventory current
- Cross-references complete
- Filing consistent
- Gaps identified and addressed

5. eTMF CONSIDERATIONS

5.1 System Requirements
- Validated per 21 CFR Part 11
- Audit trail maintained
- Access controls implemented
- Backup procedures in place

5.2 User Responsibilities
- Use correct naming conventions
- Complete all metadata
- Upload timely
- Verify upload success

6. QUALITY CONTROL
- Weekly filing compliance check
- Monthly completeness review
- Quarterly TMF audit
- Annual system validation

7. ARCHIVING
- Per retention requirements
- Secure storage
- Accessibility maintained
- Destruction procedures documented
            """,
            "keywords": ["TMF", "trial master file", "essential documents", "filing", "archiving"],
            "roles": ["Document Manager", "CRA", "Quality Assurance"],
            "related_sops": ["SOP-ES-002", "SOP-AU-001", "SOP-RE-001"]
        },
        
        {
            "sop_id": "SOP-ES-002",
            "title": "Document Retention and Archiving",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Quality Assurance",
            "category": "essential_documents",
            "objective": "To establish procedures for retaining and archiving clinical trial documents.",
            "scope": "This SOP applies to all personnel responsible for document retention and archival.",
            "content": """
1. PURPOSE
This SOP describes the procedures for retaining clinical trial documents and transferring them to archive.

2. RETENTION REQUIREMENTS

2.1 Regulatory Requirements
- ICH: 2 years after last approval or discontinuation
- FDA: 2 years after marketing approval
- EMA: 25 years after study completion
- Apply longest applicable requirement

2.2 Document Categories
- Essential documents
- Supporting documents
- Source documents (site)
- Electronic records

3. RETENTION PROCEDURES

3.1 Active Retention
- During trial conduct
- Readily accessible
- Regular review
- Update as needed

3.2 Archive Preparation
- Completeness verification
- Index creation
- Box/folder labeling
- Transfer documentation

3.3 Archive Transfer
- Secure transport
- Chain of custody
- Receipt confirmation
- Location documentation

4. ARCHIVE REQUIREMENTS

4.1 Physical Archives
- Climate controlled
- Fire/flood protection
- Access controlled
- Regular inventory

4.2 Electronic Archives
- Validated systems
- Backup procedures
- Media migration plan
- Accessibility verified

5. RETRIEVAL PROCEDURES
- Formal request required
- Authorization verified
- Retrieval documented
- Return tracked

6. DESTRUCTION PROCEDURES

6.1 Authorization
- Sponsor approval required
- Retention period verified
- Regulatory check

6.2 Process
- Destruction method appropriate
- Witnessed if required
- Certificate obtained
- Documentation updated

7. DOCUMENTATION
- Retention schedule
- Archive inventory
- Retrieval log
- Destruction certificates
            """,
            "keywords": ["retention", "archiving", "storage", "destruction", "records management"],
            "roles": ["Document Manager", "Quality Assurance", "Study Manager"],
            "related_sops": ["SOP-ES-001", "SOP-AU-001", "SOP-RE-001"]
        },
        
        # SOP 6: Audit and Compliance
        {
            "sop_id": "SOP-AU-001",
            "title": "Audit Trail Review Procedures",
            "version": "2.0",
            "effective_date": "2024-01-01",
            "department": "Quality Assurance",
            "category": "audit",
            "objective": "To establish procedures for reviewing electronic audit trails in clinical trials.",
            "scope": "This SOP applies to data management and quality assurance personnel responsible for audit trail review.",
            "content": """
1. PURPOSE
This SOP describes the procedures for reviewing audit trails to ensure data integrity and regulatory compliance.

2. AUDIT TRAIL REQUIREMENTS

2.1 Regulatory Basis
- 21 CFR Part 11
- ICH E6(R2)
- EMA Annex 11
- Regional requirements

2.2 Required Elements
- User identification
- Date and time stamp
- Previous value
- New value
- Reason for change (if applicable)

3. REVIEW FREQUENCY
- Routine: Monthly sample review
- For-cause: Upon request
- Pre-lock: Comprehensive review
- Inspection: As requested

4. REVIEW PROCEDURES

4.1 Sample Selection
- Random sample per site
- Risk-based selection
- Critical data focus
- Unusual pattern investigation

4.2 Review Activities
a) Export audit trail data
b) Review changes chronologically
c) Identify unusual patterns
d) Document findings
e) Follow up on concerns

4.3 Red Flags
- Excessive changes to same field
- Changes by unauthorized users
- Changes after database lock
- Pattern of late changes
- Unusual timing of changes

5. DOCUMENTATION
- Review date and reviewer
- Records reviewed
- Findings
- Follow-up actions
- Resolution

6. ESCALATION
- Significant findings to QA
- Data integrity concerns to management
- Potential fraud to compliance
- Regulatory implications assessed

7. TRAINING
- Reviewers trained on procedures
- Training documented
- Annual refresher required
            """,
            "keywords": ["audit trail", "review", "data integrity", "21 CFR Part 11", "compliance"],
            "roles": ["Data Manager", "Quality Assurance", "Compliance Officer"],
            "related_sops": ["SOP-AU-002", "SOP-RE-001", "SOP-DM-001"]
        },
        
        {
            "sop_id": "SOP-AU-002",
            "title": "Inspection Preparation and Response",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Quality Assurance",
            "category": "audit",
            "objective": "To establish procedures for preparing for and responding to regulatory inspections.",
            "scope": "This SOP applies to all personnel who may be involved in regulatory inspections.",
            "content": """
1. PURPOSE
This SOP describes the procedures for preparing for regulatory inspections and responding to inspection findings.

2. INSPECTION TYPES
- FDA BIMO inspection
- EMA GCP inspection
- National authority inspection
- For-cause inspection
- Pre-approval inspection

3. INSPECTION PREPARATION

3.1 Ongoing Readiness
- TMF current and complete
- Essential documents filed
- Training records updated
- Procedures followed

3.2 Pre-Inspection Activities
- Designate inspection coordinator
- Review TMF completeness
- Prepare inspection room
- Brief staff on procedures
- Prepare key documents

3.3 Staff Preparation
- Review roles and responsibilities
- Protocol refresher
- Procedure review
- Mock questions
- Communication guidelines

4. DURING INSPECTION

4.1 Logistics
- Inspection room equipped
- Refreshments available
- Copier/scanner accessible
- Contact list available

4.2 Conduct
- Professional and cooperative
- Answer questions directly
- Provide requested documents promptly
- Do not volunteer information
- Take detailed notes
- Clarify questions if unclear

4.3 Documentation
- Log all requests
- Track documents provided
- Note questions and responses
- Daily summary meetings

5. POST-INSPECTION

5.1 Close-out Meeting
- Attend all team members
- Note all observations
- Clarify any questions
- Thank inspectors

5.2 Response to Findings
- Review findings carefully
- Develop response plan
- Assign responsibilities
- Set timelines
- Draft response letter

5.3 CAPA Development
- Root cause analysis
- Corrective actions
- Preventive actions
- Implementation plan
- Effectiveness monitoring

6. DOCUMENTATION
- Inspection log
- Request/response tracking
- Finding summaries
- CAPA documentation
- Response submissions
            """,
            "keywords": ["inspection", "preparation", "response", "CAPA", "regulatory", "FDA"],
            "roles": ["Quality Assurance", "Study Manager", "Site Coordinator"],
            "related_sops": ["SOP-AU-001", "SOP-ES-001", "SOP-RE-001"]
        },
        
        # SOP 7: Regulatory
        {
            "sop_id": "SOP-RE-001",
            "title": "Regulatory Compliance Procedures",
            "version": "4.0",
            "effective_date": "2024-01-01",
            "department": "Regulatory Affairs",
            "category": "regulatory",
            "objective": "To establish procedures for maintaining regulatory compliance throughout clinical trials.",
            "scope": "This SOP applies to regulatory affairs, clinical operations, and quality assurance personnel.",
            "content": """
1. PURPOSE
This SOP describes the procedures for ensuring and maintaining regulatory compliance in clinical trial conduct.

2. REGULATORY FRAMEWORK

2.1 Key Regulations
- ICH E6(R2) GCP Guideline
- 21 CFR Parts 11, 50, 56, 312
- EU CTR 536/2014
- Regional/national requirements

2.2 Guidance Documents
- FDA Guidance for Industry
- EMA guidance documents
- ICH guidelines
- Regional guidance

3. COMPLIANCE REQUIREMENTS

3.1 Protocol Compliance
- Conduct per approved protocol
- Amendments properly approved
- Deviations documented
- Variances reported

3.2 Regulatory Submissions
- IND/CTA maintenance
- Safety reporting
- Annual reports
- Protocol amendments

3.3 IRB/IEC Compliance
- Initial approval obtained
- Continuing review
- Amendment approvals
- Safety reports submitted

3.4 GCP Compliance
- Investigator qualifications
- Informed consent
- Source documentation
- Essential documents

4. COMPLIANCE MONITORING

4.1 Internal Monitoring
- SOP compliance audits
- Process reviews
- Training verification
- Document reviews

4.2 External Monitoring
- Regulatory inspection readiness
- Third-party audits
- Sponsor audits

5. NON-COMPLIANCE HANDLING

5.1 Identification
- Self-identified
- Audit finding
- Inspection observation
- Reported issue

5.2 Assessment
- Severity determination
- Impact evaluation
- Root cause analysis
- Regulatory reporting needs

5.3 Correction
- Immediate correction
- Preventive measures
- Documentation
- Follow-up verification

6. TRAINING
- GCP training required
- Protocol-specific training
- SOP training
- Refresher training
- Documentation maintained

7. DOCUMENTATION
- Training records
- Compliance assessments
- CAPA documentation
- Regulatory correspondence
            """,
            "keywords": ["regulatory", "compliance", "GCP", "FDA", "ICH", "IRB"],
            "roles": ["Regulatory Affairs", "Quality Assurance", "Study Manager"],
            "related_sops": ["SOP-RE-002", "SOP-AU-001", "SOP-ES-001"]
        },
        
        {
            "sop_id": "SOP-RE-002",
            "title": "21 CFR Part 11 Compliance",
            "version": "2.5",
            "effective_date": "2024-01-01",
            "department": "Quality Assurance",
            "category": "regulatory",
            "objective": "To establish procedures for ensuring compliance with 21 CFR Part 11 requirements.",
            "scope": "This SOP applies to all personnel using electronic systems for clinical trial records.",
            "content": """
1. PURPOSE
This SOP describes the procedures for ensuring that electronic records and electronic signatures meet 21 CFR Part 11 requirements.

2. SCOPE OF APPLICATION

2.1 Covered Records
- Electronic records that are GxP regulated
- Electronic signatures used in lieu of handwritten
- Hybrid systems (paper + electronic)

2.2 System Categories
- EDC systems
- CTMS
- eTMF
- Safety databases
- Electronic diaries

3. REQUIREMENTS

3.1 System Validation (§11.10(a))
- Validation protocols
- Testing documentation
- Validation reports
- Change control

3.2 Copies of Records (§11.10(b))
- Accurate copies generated
- Human-readable format
- Electronic format available

3.3 Record Protection (§11.10(c))
- Access controls
- Backup procedures
- Disaster recovery

3.4 Limiting Access (§11.10(d))
- Authorized users only
- Role-based access
- Access reviews
- Deactivation procedures

3.5 Audit Trail (§11.10(e))
- Computer-generated
- Time-stamped
- Previous values maintained
- Independent of operators

3.6 Sequence Checks (§11.10(f))
- Operational steps followed
- Sequencing verified

3.7 Authority Checks (§11.10(g))
- User authorization verified
- Permission levels enforced

3.8 Device Checks (§11.10(h))
- Source validity verified

3.9 Training (§11.10(i))
- Personnel trained
- Training documented
- Refresher training

3.10 Written Policies (§11.10(j))
- Signature accountability
- Electronic signature policy

3.11 System Documentation (§11.10(k))
- Controls documented
- Available for inspection

4. ELECTRONIC SIGNATURES

4.1 Requirements (§11.100)
- Unique to individual
- Identity verified
- Not reused/reassigned

4.2 Signature Components (§11.200)
- Identification component
- Password component
- Continuous session signing

4.3 Signature Manifestation (§11.50)
- Printed name
- Date and time
- Meaning of signature

5. COMPLIANCE PROCEDURES
- System assessment
- Gap analysis
- Remediation
- Ongoing monitoring

6. DOCUMENTATION
- Validation documentation
- Training records
- Procedure documentation
- Audit trail access
            """,
            "keywords": ["21 CFR Part 11", "electronic records", "electronic signatures", "validation", "audit trail"],
            "roles": ["IT", "Quality Assurance", "Data Management"],
            "related_sops": ["SOP-RE-001", "SOP-AU-001", "SOP-DM-001"]
        },
        
        # SOP 8: Medical Coding
        {
            "sop_id": "SOP-CD-001",
            "title": "Medical Coding Procedures",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "department": "Data Management",
            "category": "coding",
            "objective": "To establish procedures for coding adverse events and medical history using MedDRA.",
            "scope": "This SOP applies to medical coders and data management personnel involved in AE/MH coding.",
            "content": """
1. PURPOSE
This SOP describes the procedures for coding adverse events and medical history terms using the Medical Dictionary for Regulatory Activities (MedDRA).

2. CODING DICTIONARIES

2.1 MedDRA
- Used for AEs, medical history, indications
- Version controlled per study
- Hierarchy: SOC > HLGT > HLT > PT > LLT

2.2 Dictionary Version
- Specified in Data Management Plan
- Version locked during study
- Upgrades per defined process

3. CODING PROCEDURES

3.1 Verbatim Term Review
a) Review verbatim term from CRF
b) Assess completeness and clarity
c) Query site if unclear
d) Document clarification

3.2 Coding Assignment
a) Search dictionary for matching term
b) Select most specific appropriate term
c) Code to LLT/PT level
d) Verify SOC assignment
e) Document coding decision

3.3 Coding Guidelines
- Code to reported term, not diagnosis
- Use most specific term available
- Avoid assumptions
- Query if uncertain
- Document rationale

4. CODING LEVELS

4.1 Preferred Term (PT)
- Primary coding level for analysis
- Represents single medical concept
- Used in safety tables

4.2 Lowest Level Term (LLT)
- Most granular level
- Linked to one PT
- Used for verbatim matching

4.3 System Organ Class (SOC)
- Highest grouping level
- Used for reporting
- Primary SOC assigned

5. AUTO-CODING
- System suggests codes for common terms
- Coder verifies suggestions
- Acceptance or override documented
- High-confidence threshold defined

6. QUALITY CONTROL
- Daily review of coded terms
- Batch review before lock
- Consistency checks
- Inter-coder reliability

7. DOCUMENTATION
- Coding conventions document
- Decision log
- Query documentation
- Version history
            """,
            "keywords": ["MedDRA", "coding", "adverse events", "medical history", "PT", "LLT"],
            "roles": ["Medical Coder", "Data Manager"],
            "related_sops": ["SOP-CD-002", "SOP-DM-001", "SOP-SA-001"]
        },
        
        {
            "sop_id": "SOP-CD-002",
            "title": "Drug Coding Procedures",
            "version": "2.5",
            "effective_date": "2024-01-01",
            "department": "Data Management",
            "category": "coding",
            "objective": "To establish procedures for coding concomitant medications using WHODrug.",
            "scope": "This SOP applies to medical coders and data management personnel involved in drug coding.",
            "content": """
1. PURPOSE
This SOP describes the procedures for coding concomitant medications and prior therapies using the WHODrug dictionary.

2. CODING DICTIONARY

2.1 WHODrug Global
- Used for medications
- Version controlled per study
- Contains drug names, ingredients, ATC codes

2.2 Dictionary Version
- Specified in Data Management Plan
- Version locked during study
- Updates per defined process

3. CODING PROCEDURES

3.1 Drug Name Review
a) Review drug name from CRF
b) Assess completeness (name, formulation, route)
c) Query site if incomplete
d) Consider regional variations

3.2 Coding Assignment
a) Search dictionary for drug name
b) Match on trade name or generic
c) Verify formulation match
d) Assign ATC code
e) Document coding

3.3 Coding Guidelines
- Match trade name when available
- Use generic for combination products
- Consider regional brand names
- Query for clarification if needed

4. CODING LEVELS

4.1 Drug Record
- Drug name
- Formulation
- Route of administration
- Manufacturer (if needed)

4.2 ATC Classification
- Anatomical main group
- Therapeutic subgroup
- Pharmacological subgroup
- Chemical subgroup
- Chemical substance

5. HANDLING SPECIAL CASES

5.1 Combination Products
- Code to appropriate combination record
- Or code individual ingredients

5.2 Herbal/OTC Products
- Use WHODrug when available
- Document if not in dictionary

5.3 Investigational Drugs
- Do not code study drug
- Code other investigational per plan

6. QUALITY CONTROL
- Daily coding review
- Batch review before lock
- ATC code verification
- Inter-coder reliability checks

7. DOCUMENTATION
- Coding conventions
- Decision log
- Query documentation
- Version history
            """,
            "keywords": ["WHODrug", "drug coding", "concomitant medications", "ATC", "pharmaceutical"],
            "roles": ["Medical Coder", "Data Manager"],
            "related_sops": ["SOP-CD-001", "SOP-DM-001"]
        }
    ]
}


def get_all_sops() -> dict:
    """Return all SOP documents."""
    return SOP_DOCUMENTS


def get_sop_by_id(sop_id: str) -> dict:
    """Get a specific SOP by ID."""
    for sop in SOP_DOCUMENTS["sops"]:
        if sop["sop_id"] == sop_id:
            return sop
    return None


def get_sops_by_category(category: str) -> list:
    """Get all SOPs in a category."""
    return [s for s in SOP_DOCUMENTS["sops"] if s["category"] == category]


def get_sops_by_department(department: str) -> list:
    """Get all SOPs by department."""
    return [s for s in SOP_DOCUMENTS["sops"] if s["department"] == department]


def get_all_categories() -> list:
    """Get list of all SOP categories."""
    categories = set()
    for sop in SOP_DOCUMENTS["sops"]:
        categories.add(sop["category"])
    return sorted(list(categories))


def get_all_departments() -> list:
    """Get list of all departments."""
    departments = set()
    for sop in SOP_DOCUMENTS["sops"]:
        departments.add(sop["department"])
    return sorted(list(departments))


def search_sops(query: str) -> list:
    """Search SOPs by query string."""
    query_lower = query.lower()
    results = []
    
    for sop in SOP_DOCUMENTS["sops"]:
        score = 0
        
        # Title match (highest weight)
        if query_lower in sop["title"].lower():
            score += 10
        
        # Keyword match (high weight)
        for keyword in sop.get("keywords", []):
            if query_lower in keyword.lower():
                score += 5
        
        # Objective match
        if query_lower in sop.get("objective", "").lower():
            score += 3
        
        # Content match (lower weight)
        if query_lower in sop["content"].lower():
            score += 1
        
        if score > 0:
            results.append({
                "sop_id": sop["sop_id"],
                "title": sop["title"],
                "category": sop["category"],
                "department": sop["department"],
                "relevance_score": score,
                "keywords": sop.get("keywords", [])
            })
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


def get_sop_summary() -> dict:
    """Get summary statistics of SOPs."""
    categories = {}
    departments = {}
    
    for sop in SOP_DOCUMENTS["sops"]:
        cat = sop["category"]
        dept = sop["department"]
        
        categories[cat] = categories.get(cat, 0) + 1
        departments[dept] = departments.get(dept, 0) + 1
    
    return {
        "total_sops": len(SOP_DOCUMENTS["sops"]),
        "categories": categories,
        "departments": departments,
        "version": SOP_DOCUMENTS["document_info"]["version"]
    }
