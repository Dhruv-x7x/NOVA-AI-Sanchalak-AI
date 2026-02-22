# File: src/knowledge/documents/ich_gcp_guidelines.py
"""
ICH-GCP Guidelines Knowledge Base
Complete E6(R2) Guidelines for Clinical Trial Conduct
"""

ICH_GCP_GUIDELINES = {
    "document_info": {
        "title": "ICH Harmonised Guideline - Integrated Addendum to ICH E6(R1): Guideline for Good Clinical Practice E6(R2)",
        "version": "E6(R2)",
        "effective_date": "2016-11-09",
        "source": "International Council for Harmonisation"
    },
    
    "sections": [
        # Section 1: Glossary
        {
            "section_id": "1",
            "title": "Glossary",
            "content": """
1.1 Adverse Drug Reaction (ADR): In the pre-approval clinical experience with a new medicinal product or its new usages, particularly as the therapeutic dose(s) may not be established: all noxious and unintended responses to a medicinal product related to any dose should be considered adverse drug reactions.

1.2 Adverse Event (AE): Any untoward medical occurrence in a patient or clinical investigation subject administered a pharmaceutical product and which does not necessarily have a causal relationship with this treatment.

1.3 Amendment (to the protocol): A written description of a change(s) to or formal clarification of a protocol.

1.4 Applicable Regulatory Requirement(s): Any law(s) and regulation(s) addressing the conduct of clinical trials of investigational products.

1.5 Approval (in relation to Institutional Review Boards): The affirmative decision of the IRB that the clinical trial has been reviewed and may be conducted at the institution site within the constraints set forth by the IRB, the institution, Good Clinical Practice (GCP), and the applicable regulatory requirements.

1.6 Audit: A systematic and independent examination of trial related activities and documents to determine whether the evaluated trial related activities were conducted, and the data were recorded, analyzed and accurately reported according to the protocol, sponsor's standard operating procedures (SOPs), Good Clinical Practice (GCP), and the applicable regulatory requirement(s).

1.7 Audit Certificate: A declaration of confirmation by the auditor that an audit has taken place.

1.8 Audit Report: A written evaluation by the sponsor's auditor of the results of the audit.

1.9 Audit Trail: Documentation that allows reconstruction of the course of events.

1.10 Blinding/Masking: A procedure in which one or more parties to the trial are kept unaware of the treatment assignment(s).

1.11 Case Report Form (CRF): A printed, optical, or electronic document designed to record all of the protocol required information to be reported to the sponsor on each trial subject.

1.12 Clinical Trial/Study: Any investigation in human subjects intended to discover or verify the clinical, pharmacological and/or other pharmacodynamic effects of an investigational product(s).

1.13 Comparator (Product): An investigational or marketed product (i.e., active control), or placebo, used as a reference in a clinical trial.

1.14 Compliance (in relation to trials): Adherence to all the trial-related requirements, Good Clinical Practice (GCP) requirements, and the applicable regulatory requirements.

1.15 Confidentiality: Prevention of disclosure, to other than authorized individuals, of a sponsor's proprietary information or of a subject's identity.

1.16 Contract: A written, dated, and signed agreement between two or more involved parties that sets out any arrangements on delegation and distribution of tasks and obligations.

1.17 Coordinating Committee: A committee that a sponsor may organize to coordinate the conduct of a multicentre trial.

1.18 Coordinating Investigator: An investigator assigned the responsibility for the coordination of investigators at different centres participating in a multicentre trial.

1.19 Contract Research Organization (CRO): A person or an organization (commercial, academic, or other) contracted by the sponsor to perform one or more of a sponsor's trial-related duties and functions.

1.20 Direct Access: Permission to examine, analyze, verify, and reproduce any records and reports that are important to evaluation of a clinical trial.

1.21 Documentation: All records, in any form (including, but not limited to, written, electronic, magnetic, and optical records, and scans, x-rays, and electrocardiograms) that describe or record the methods, conduct, and/or results of a trial, the factors affecting a trial, and the actions taken.

1.22 Essential Documents: Documents which individually and collectively permit evaluation of the conduct of a study and the quality of the data produced.

1.23 Good Clinical Practice (GCP): A standard for the design, conduct, performance, monitoring, auditing, recording, analyses, and reporting of clinical trials that provides assurance that the data and reported results are credible and accurate, and that the rights, integrity, and confidentiality of trial subjects are protected.

1.24 Independent Data-Monitoring Committee (IDMC): An independent data-monitoring committee that may be established by the sponsor to assess at intervals the progress of a clinical trial, the safety data, and the critical efficacy endpoints.

1.25 Impartial Witness: A person, who is independent of the trial, who cannot be unfairly influenced by people involved with the trial.

1.26 Independent Ethics Committee (IEC): An independent body (a review board or a committee, institutional, regional, national, or supranational), constituted of medical professionals and non-medical members.

1.27 Informed Consent: A process by which a subject voluntarily confirms his or her willingness to participate in a particular trial, after having been informed of all aspects of the trial that are relevant to the subject's decision to participate.

1.28 Inspection: The act by a regulatory authority(ies) of conducting an official review of documents, facilities, records, and any other resources that are deemed by the authority(ies) to be related to the clinical trial.

1.29 Institution (medical): Any public or private entity or agency or medical or dental facility where clinical trials are conducted.

1.30 Institutional Review Board (IRB): An independent body constituted of medical, scientific, and non-scientific members, whose responsibility is to ensure the protection of the rights, safety and well-being of human subjects involved in a trial.

1.31 Interim Clinical Trial/Study Report: A report of intermediate results and their evaluation based on analyses performed during the course of a trial.

1.32 Investigational Product: A pharmaceutical form of an active ingredient or placebo being tested or used as a reference in a clinical trial.

1.33 Investigator: A person responsible for the conduct of the clinical trial at a trial site.

1.34 Investigator / Institution: An expression meaning "the investigator and/or institution, where required by the applicable regulatory requirements".

1.35 Investigator's Brochure: A compilation of the clinical and nonclinical data on the investigational product(s) which is relevant to the study of the investigational product(s) in human subjects.

1.36 Legally Acceptable Representative: An individual or juridical or other body authorized under applicable law to consent, on behalf of a prospective subject, to the subject's participation in the clinical trial.

1.37 Monitoring: The act of overseeing the progress of a clinical trial, and of ensuring that it is conducted, recorded, and reported in accordance with the protocol, Standard Operating Procedures (SOPs), Good Clinical Practice (GCP), and the applicable regulatory requirement(s).

1.38 Monitoring Report: A written report from the monitor to the sponsor after each site visit and/or other trial-related communication according to the sponsor's SOPs.

1.39 Multicentre Trial: A clinical trial conducted according to a single protocol but at more than one site.

1.40 Nonclinical Study: Biomedical studies not performed on human subjects.

1.41 Opinion (in relation to Independent Ethics Committee): The judgement and/or the advice provided by an Independent Ethics Committee (IEC).

1.42 Original Medical Record: See Source Documents.

1.43 Protocol: A document that describes the objective(s), design, methodology, statistical considerations, and organization of a trial.

1.44 Protocol Amendment: See Amendment.

1.45 Quality Assurance (QA): All those planned and systematic actions that are established to ensure that the trial is performed and the data are generated, documented (recorded), and reported in compliance with Good Clinical Practice (GCP) and the applicable regulatory requirement(s).

1.46 Quality Control (QC): The operational techniques and activities undertaken within the quality assurance system to verify that the requirements for quality of the trial-related activities have been fulfilled.

1.47 Randomization: The process of assigning trial subjects to treatment or control groups using an element of chance to determine the assignments in order to reduce bias.

1.48 Regulatory Authorities: Bodies having the power to regulate. In the ICH GCP guideline the expression Regulatory Authorities includes the authorities that review submitted clinical data and those that conduct inspections.

1.49 Serious Adverse Event (SAE) or Serious Adverse Drug Reaction (Serious ADR): Any untoward medical occurrence that at any dose results in death, is life-threatening, requires inpatient hospitalization or prolongation of existing hospitalization, results in persistent or significant disability/incapacity, or is a congenital anomaly/birth defect.

1.50 Source Data: All information in original records and certified copies of original records of clinical findings, observations, or other activities in a clinical trial necessary for the reconstruction and evaluation of the trial.

1.51 Source Documents: Original documents, data, and records (e.g., hospital records, clinical and office charts, laboratory notes, memoranda, subjects' diaries or evaluation checklists, pharmacy dispensing records, recorded data from automated instruments, copies or transcriptions certified after verification as being accurate copies, microfiches, photographic negatives, microfilm or magnetic media, x-rays, subject files, and records kept at the pharmacy, at the laboratories and at medico-technical departments involved in the clinical trial).

1.52 Sponsor: An individual, company, institution, or organization which takes responsibility for the initiation, management, and/or financing of a clinical trial.

1.53 Sponsor-Investigator: An individual who both initiates and conducts, alone or with others, a clinical trial, and under whose immediate direction the investigational product is administered to, dispensed to, or used by a subject.

1.54 Standard Operating Procedures (SOPs): Detailed, written instructions to achieve uniformity of the performance of a specific function.

1.55 Subinvestigator: Any individual member of the clinical trial team designated and supervised by the investigator at a trial site to perform critical trial-related procedures and/or to make important trial-related decisions.

1.56 Subject/Trial Subject: An individual who participates in a clinical trial, either as a recipient of the investigational product(s) or as a control.

1.57 Subject Identification Code: A unique identifier assigned by the investigator to each trial subject to protect the subject's identity and used in lieu of the subject's name when the investigator reports adverse events and/or other trial related data.

1.58 Trial Master File: The essential documents that individually and collectively permit evaluation of the conduct of a trial and the quality of the data produced.

1.59 Trial Site: The location(s) where trial-related activities are actually conducted.

1.60 Unexpected Adverse Drug Reaction: An adverse reaction, the nature or severity of which is not consistent with the applicable product information.

1.61 Vulnerable Subjects: Individuals whose willingness to volunteer in a clinical trial may be unduly influenced by the expectation, whether justified or not, of benefits associated with participation, or of a retaliatory response from senior members of a hierarchy in case of refusal to participate.

1.62 Well-being (of the trial subjects): The physical and mental integrity of the subjects participating in a clinical trial.
            """,
            "subsections": []
        },
        
        # Section 2: Principles of ICH GCP
        {
            "section_id": "2",
            "title": "The Principles of ICH GCP",
            "content": """
2.1 Clinical trials should be conducted in accordance with the ethical principles that have their origin in the Declaration of Helsinki, and that are consistent with GCP and the applicable regulatory requirement(s).

2.2 Before a trial is initiated, foreseeable risks and inconveniences should be weighed against the anticipated benefit for the individual trial subject and society. A trial should be initiated and continued only if the anticipated benefits justify the risks.

2.3 The rights, safety, and well-being of the trial subjects are the most important considerations and should prevail over interests of science and society.

2.4 The available nonclinical and clinical information on an investigational product should be adequate to support the proposed clinical trial.

2.5 Clinical trials should be scientifically sound, and described in a clear, detailed protocol.

2.6 A trial should be conducted in compliance with the protocol that has received prior institutional review board (IRB)/independent ethics committee (IEC) approval/favourable opinion.

2.7 The medical care given to, and medical decisions made on behalf of, subjects should always be the responsibility of a qualified physician or, when appropriate, of a qualified dentist.

2.8 Each individual involved in conducting a trial should be qualified by education, training, and experience to perform his or her respective task(s).

2.9 Freely given informed consent should be obtained from every subject prior to clinical trial participation.

2.10 All clinical trial information should be recorded, handled, and stored in a way that allows its accurate reporting, interpretation and verification.

2.11 The confidentiality of records that could identify subjects should be protected, respecting the privacy and confidentiality rules in accordance with the applicable regulatory requirement(s).

2.12 Investigational products should be manufactured, handled, and stored in accordance with applicable good manufacturing practice (GMP). They should be used in accordance with the approved protocol.

2.13 Systems with procedures that assure the quality of every aspect of the trial should be implemented.

2.14 ADDENDUM: Throughout the clinical trial, the sponsor is responsible for implementing and maintaining quality assurance and quality control systems with written SOPs to ensure that trials are conducted and data are generated, documented, and reported in compliance with the protocol, GCP, and the applicable regulatory requirements.

2.15 ADDENDUM: Throughout the clinical trial, the sponsor should ensure appropriate safeguards are in place to protect the confidentiality of subject data in compliance with the applicable regulatory requirements.
            """,
            "subsections": []
        },
        
        # Section 3: IRB/IEC
        {
            "section_id": "3",
            "title": "Institutional Review Board/Independent Ethics Committee (IRB/IEC)",
            "content": """
3.1 Responsibilities
3.1.1 An IRB/IEC should safeguard the rights, safety, and well-being of all trial subjects. Special attention should be paid to trials that may include vulnerable subjects.

3.1.2 The IRB/IEC should obtain the following documents: trial protocol(s)/amendment(s), written informed consent form(s), consent form updates, subject recruitment procedures, written information to be provided to subjects, Investigator's Brochure (IB), available safety information, information about payments and compensation available to subjects, the investigator's current curriculum vitae and/or other documentation evidencing qualifications, and any other documents that the IRB/IEC may need to fulfil its responsibilities.

3.1.3 The IRB/IEC should review a proposed clinical trial within a reasonable time and document its views in writing, clearly identifying the trial, the documents reviewed and the dates for approval, modifications required, disapproval, or termination.

3.1.4 The IRB/IEC should consider the qualifications of the investigator for the proposed trial, as documented by a current curriculum vitae and/or by any other relevant documentation the IRB/IEC requests.

3.1.5 The IRB/IEC should conduct continuing review of each ongoing trial at intervals appropriate to the degree of risk to human subjects, but at least once per year.

3.1.6 The IRB/IEC may request more information than is outlined in paragraph 3.1.2 be given to subjects when, in the judgement of the IRB/IEC, the additional information would add meaningfully to the protection of the rights, safety and/or well-being of the subjects.

3.1.7 When a non-therapeutic trial is to be carried out with the consent of the subject's legally acceptable representative, the IRB/IEC should determine that the proposed protocol and/or other document(s) adequately addresses relevant ethical concerns and meets applicable regulatory requirements for such trials.

3.1.8 Where the protocol indicates that prior consent of the trial subject or the subject's legally acceptable representative is not possible, the IRB/IEC should determine that the proposed protocol and/or other document(s) adequately addresses relevant ethical concerns and meets applicable regulatory requirements for such trials.

3.1.9 The IRB/IEC should review both the amount and method of payment to subjects to assure that neither presents problems of coercion or undue influence on the trial subjects.

3.2 Composition, Functions and Operations
3.2.1 The IRB/IEC should consist of a reasonable number of members, who collectively have the qualifications and experience to review and evaluate the science, medical aspects, and ethics of the proposed trial.

3.2.2 The IRB/IEC should include at least one member whose primary area of interest is in a nonscientific area and at least one member who is independent of the institution/trial site.

3.2.3 Only those IRB/IEC members who are independent of the investigator and the sponsor of the trial should vote/provide opinion on a trial-related matter.

3.2.4 The IRB/IEC should perform its functions according to written operating procedures, maintain written records of its activities and minutes of its meetings, and comply with GCP and with the applicable regulatory requirement(s).

3.2.5 An IRB/IEC should make its decisions at announced meetings at which at least a quorum, as stipulated in its written operating procedures, is present.

3.2.6 Only members who participate in the IRB/IEC review and discussion should vote/provide their opinion and/or advise.

3.2.7 The investigator may provide information on any aspect of the trial, but should not participate in the deliberations of the IRB/IEC or in the vote/opinion of the IRB/IEC.

3.2.8 An IRB/IEC may invite nonmembers with expertise in special areas for assistance.

3.3 Procedures
The IRB/IEC should establish, document in writing, and follow its procedures.

3.4 Records
The IRB/IEC should retain all relevant records for a period of at least 3 years after completion of the trial.
            """,
            "subsections": []
        },
        
        # Section 4: Investigator
        {
            "section_id": "4",
            "title": "Investigator",
            "content": """
4.1 Investigator's Qualifications and Agreements
4.1.1 The investigator(s) should be qualified by education, training, and experience to assume responsibility for the proper conduct of the trial.

4.1.2 The investigator should be thoroughly familiar with the appropriate use of the investigational product(s), as described in the protocol, in the current Investigator's Brochure, in the product information and in other information sources provided by the sponsor.

4.1.3 The investigator should be aware of, and should comply with, GCP and the applicable regulatory requirements.

4.1.4 The investigator/institution should permit monitoring and auditing by the sponsor, and inspection by the appropriate regulatory authority(ies).

4.1.5 The investigator should maintain a list of appropriately qualified persons to whom the investigator has delegated significant trial-related duties.

4.2 Adequate Resources
4.2.1 The investigator should be able to demonstrate (e.g., based on retrospective data) a potential for recruiting the required number of suitable subjects within the agreed recruitment period.

4.2.2 The investigator should have sufficient time to properly conduct and complete the trial within the agreed trial period.

4.2.3 The investigator should have available an adequate number of qualified staff and adequate facilities for the foreseen duration of the trial.

4.2.4 The investigator should ensure that all persons assisting with the trial are adequately informed about the protocol, the investigational product(s), and their trial-related duties and functions.

4.3 Medical Care of Trial Subjects
4.3.1 A qualified physician (or dentist, when appropriate), who is an investigator or a sub-investigator for the trial, should be responsible for all trial-related medical (or dental) decisions.

4.3.2 During and following a subject's participation in a trial, the investigator/institution should ensure that adequate medical care is provided to a subject for any adverse events, including clinically significant laboratory values, related to the trial.

4.3.3 The investigator should inform a subject when medical care is needed for intercurrent illness(es) of which the investigator becomes aware.

4.3.4 It is recommended that the investigator inform the subject's primary physician about the subject's participation in the trial if the subject has a primary physician and if the subject agrees to the primary physician being informed.

4.4 Communication with IRB/IEC
4.4.1 Before initiating a trial, the investigator/institution should have written and dated approval/favourable opinion from the IRB/IEC for the trial protocol, written informed consent form, consent form updates, subject recruitment procedures, and any other written information to be provided to subjects.

4.4.2 As part of the investigator's/institution's written application to the IRB/IEC, the investigator/institution should provide the IRB/IEC with a current copy of the Investigator's Brochure.

4.4.3 During the trial the investigator/institution should provide to the IRB/IEC all documents subject to review.

4.5 Compliance with Protocol
4.5.1 The investigator/institution should conduct the trial in compliance with the protocol agreed to by the sponsor and, if required, by the regulatory authority(ies) and which was given approval/favourable opinion by the IRB/IEC.

4.5.2 The investigator should not implement any deviation from, or changes of the protocol without agreement by the sponsor and prior review and documented approval/favourable opinion from the IRB/IEC of an amendment, except where necessary to eliminate an immediate hazard(s) to trial subjects.

4.5.3 The investigator, or person designated by the investigator, should document and explain any deviation from the approved protocol.

4.5.4 The investigator may implement a deviation from, or a change of, the protocol to eliminate an immediate hazard(s) to trial subjects without prior IRB/IEC approval/favourable opinion.

4.6 Investigational Product(s)
4.6.1 Responsibility for investigational product(s) accountability at the trial site(s) rests with the investigator/institution.

4.6.2 Where allowed/required, the investigator/institution may/should assign some or all of the investigator's/institution's duties for investigational product(s) accountability at the trial site(s) to an appropriate pharmacist or another appropriate individual who is under the supervision of the investigator/institution.

4.6.3 The investigator/institution and/or a pharmacist or other appropriate individual, who is designated by the investigator/institution, should maintain records of the product's delivery to the trial site, the inventory at the site, the use by each subject, and the return to the sponsor or alternative disposition of unused product(s).

4.6.4 The investigational product(s) should be stored as specified by the sponsor and in accordance with applicable regulatory requirement(s).

4.6.5 The investigator should ensure that the investigational product(s) are used only in accordance with the approved protocol.

4.6.6 The investigator, or a person designated by the investigator/institution, should explain the correct use of the investigational product(s) to each subject and should check, at intervals appropriate for the trial, that each subject is following the instructions properly.

4.7 Randomization Procedures and Unblinding
The investigator should follow the trial's randomization procedures, if any, and should ensure that the code is broken only in accordance with the protocol.

4.8 Informed Consent of Trial Subjects
4.8.1 In obtaining and documenting informed consent, the investigator should comply with the applicable regulatory requirement(s), and should adhere to GCP and to the ethical principles that have their origin in the Declaration of Helsinki.

4.8.2 Before informed consent may be obtained, the investigator, or a person designated by the investigator, should provide the subject or the subject's legally acceptable representative ample time and opportunity to inquire about details of the trial and to decide whether or not to participate in the trial.

4.8.3 Neither the investigator, nor the trial staff, should coerce or unduly influence a subject to participate or to continue to participate in a trial.

4.8.4 None of the oral and written information concerning the trial, including the written informed consent form, should contain any language that causes the subject or the subject's legally acceptable representative to waive or to appear to waive any legal rights.

4.8.5 The investigator, or a person designated by the investigator, should fully inform the subject or, if the subject is unable to provide informed consent, the subject's legally acceptable representative, of all pertinent aspects of the trial.

4.8.6 The language used in the oral and written information about the trial, including the written informed consent form, should be as non-technical as practical.

4.8.7 If the subject or the subject's legally acceptable representative is unable to read, an impartial witness should be present during the entire informed consent discussion.

4.8.8 Before a subject's participation in the trial, the written informed consent form should be signed and personally dated by the subject or by the subject's legally acceptable representative, and by the person who conducted the informed consent discussion.

4.8.9 If a subject is unable to read or if a legally acceptable representative is unable to read, an impartial witness should be present during the entire informed consent discussion.

4.8.10 Both the informed consent discussion and the written informed consent form and any other written information to be provided to subjects should include explanations of the nature and purpose of the trial, trial treatment(s), probability of random assignment, trial procedures, subject's responsibilities, the reasonably foreseeable risks, the expected benefits, alternative procedures or treatments, compensation, confidentiality provisions, voluntary participation, and whom to contact for trial-related questions.

4.9 Records and Reports
4.9.1 The investigator should ensure the accuracy, completeness, legibility, and timeliness of the data reported to the sponsor in the CRFs and in all required reports.

4.9.2 Data reported on the CRF, that are derived from source documents, should be consistent with the source documents or the discrepancies should be explained.

4.9.3 Any change or correction to a CRF should be dated, initialed, and explained (if necessary) and should not obscure the original entry.

4.9.4 The investigator/institution should maintain the trial documents as specified in Essential Documents for the Conduct of a Clinical Trial and as required by the applicable regulatory requirement(s).

4.9.5 Essential documents should be retained until at least 2 years after the last approval of a marketing application in an ICH region and until there are no pending or contemplated marketing applications in an ICH region or at least 2 years have elapsed since the formal discontinuation of clinical development of the investigational product.

4.9.6 The financial aspects of the trial should be documented in an agreement between the sponsor and the investigator/institution.

4.9.7 Upon request of the monitor, auditor, IRB/IEC, or regulatory authority, the investigator/institution should make available for direct access all requested trial-related records.

4.10 Progress Reports
4.10.1 The investigator should submit written summaries of the trial status to the IRB/IEC annually, or more frequently, if requested by the IRB/IEC.

4.10.2 The investigator should promptly provide written reports to the sponsor, the IRB/IEC, and, where applicable, the institution on any changes significantly affecting the conduct of the trial, and/or increasing the risk to subjects.

4.11 Safety Reporting
4.11.1 All serious adverse events (SAEs) should be reported immediately to the sponsor except for those SAEs that the protocol or other document (e.g., Investigator's Brochure) identifies as not needing immediate reporting.

4.11.2 The investigator should comply with the applicable regulatory requirement(s) related to the reporting of unexpected serious adverse drug reactions to the regulatory authority(ies) and the IRB/IEC.

4.11.3 Adverse events and/or laboratory abnormalities identified in the protocol as critical to safety evaluations should be reported to the sponsor according to the reporting requirements and within the time periods specified by the sponsor in the protocol.

4.11.4 For reported deaths, the investigator should supply the sponsor and the IRB/IEC with any additional requested information.

4.12 Premature Termination or Suspension of a Trial
If the trial is prematurely terminated or suspended for any reason, the investigator/institution should promptly inform the trial subjects, should assure appropriate therapy and follow-up for the subjects, and, where required by the applicable regulatory requirement(s), should inform the regulatory authority(ies).

4.13 Final Report(s) by Investigator
Upon completion of the trial, the investigator, where applicable, should inform the institution; the investigator/institution should provide the IRB/IEC with a summary of the trial's outcome, and the regulatory authority(ies) with any reports required.
            """,
            "subsections": []
        },
        
        # Section 5: Sponsor
        {
            "section_id": "5",
            "title": "Sponsor",
            "content": """
5.0 Quality Management
5.0.1 The sponsor should implement a system to manage quality throughout all stages of the trial process.

5.0.2 Sponsors should focus on trial activities essential to ensuring human subject protection and the reliability of trial results.

5.0.3 Quality management includes the design of trial protocols and tools and procedures for the collection and processing of information, as well as the collection of information that is essential to decision making.

5.0.4 The methods used to assure and control the quality of the trial should be proportionate to the risks inherent in the trial and the importance of the information collected.

5.0.5 The sponsor should ensure that each individual involved in the trial is qualified by education, training, and experience to perform the respective task(s).

5.0.6 Quality control should be applied to each stage of data handling to ensure that all data are reliable and have been processed correctly.

5.0.7 Agreements should be documented between all involved parties to establish their roles and responsibilities.

5.1 Quality Assurance and Quality Control
5.1.1 The sponsor is responsible for implementing and maintaining quality assurance and quality control systems with written SOPs to ensure that trials are conducted and data are generated, documented (recorded), and reported in compliance with the protocol, GCP, and the applicable regulatory requirement(s).

5.1.2 The sponsor is responsible for securing agreement from all involved parties to ensure direct access to all trial related sites, source data/documents, and reports for the purpose of monitoring and auditing by the sponsor, and inspection by domestic and foreign regulatory authorities.

5.1.3 Quality control should be applied to each stage of data handling to ensure that all data are reliable and have been processed correctly.

5.1.4 Agreements, made by the sponsor with the investigator/institution and any other parties involved with the clinical trial, should be in writing, as part of the protocol or in a separate agreement.

5.2 Contract Research Organization (CRO)
5.2.1 A sponsor may transfer any or all of the sponsor's trial-related duties and functions to a CRO, but the ultimate responsibility for the quality and integrity of the trial data always resides with the sponsor.

5.2.2 Any trial-related duty and function that is transferred to and assumed by a CRO should be specified in writing.

5.2.3 Any trial-related duties and functions not specifically transferred to and assumed by a CRO are retained by the sponsor.

5.2.4 All references to a sponsor in this guideline also apply to a CRO to the extent that a CRO has assumed the trial related duties and functions of a sponsor.

5.3 Medical Expertise
The sponsor should designate appropriately qualified medical personnel who will be readily available to advise on trial related medical questions or problems.

5.4 Trial Design
5.4.1 The sponsor should utilize qualified individuals (e.g., biostatisticians, clinical pharmacologists, and physicians) as appropriate, throughout all stages of the trial process.

5.4.2 Guidance on the design of protocols is available in ICH Guideline E9 Statistical Principles for Clinical Trials.

5.5 Trial Management, Data Handling, and Record Keeping
5.5.1 The sponsor should utilize appropriately qualified individuals to supervise the overall conduct of the trial, to handle the data, to verify the data, to conduct the statistical analyses, and to prepare the trial reports.

5.5.2 The sponsor may consider establishing an independent data-monitoring committee (IDMC) to assess the progress of a clinical trial.

5.5.3 When using electronic trial data handling and/or remote electronic trial data systems, the sponsor should ensure and document that the electronic data processing system(s) conforms to the sponsor's established requirements for completeness, accuracy, reliability, and consistent intended performance.

5.5.4 The sponsor should use an unambiguous subject identification code that allows identification of all the data reported for each subject.

5.5.5 The sponsor, or other owners of the data, should retain all of the sponsor-specific essential documents pertaining to the trial.

5.5.6 The sponsor should retain all sponsor-specific essential documents in conformance with the applicable regulatory requirement(s) of the country(ies) where the product is approved.

5.5.7 If the sponsor discontinues the clinical development of an investigational product, the sponsor should maintain all sponsor-specific essential documents for at least 2 years after formal discontinuation.

5.5.8 If the sponsor discontinues the clinical development of an investigational product, the sponsor should notify all the trial investigators/institutions and all the regulatory authorities.

5.5.9 Any transfer of ownership of the data should be reported to the appropriate authority(ies).

5.5.10 The sponsor specific essential documents should be retained until at least 2 years after the last approval of a marketing application in an ICH region.

5.5.11 The sponsor should ensure the retention of records that are located at a trial site.

5.5.12 The sponsor should inform the investigator(s)/institution(s) in writing of the need for record retention and should notify the investigator(s)/institution(s) in writing when the trial related records are no longer needed.

5.6 Investigator Selection
5.6.1 The sponsor is responsible for selecting the investigator(s)/institution(s). Each investigator should be qualified by training and experience and should have adequate resources to properly conduct the trial.

5.6.2 Before entering an agreement with an investigator/institution to conduct a trial, the sponsor should provide the investigator(s)/institution(s) with the protocol and an up-to-date Investigator's Brochure.

5.6.3 The sponsor should obtain the investigator's/institution's agreement to conduct the trial in compliance with GCP, to comply with procedures for data recording and reporting, and to permit monitoring, auditing, and inspection.

5.6.4 The sponsor should obtain from the investigator/institution a signed investigator's agreement.

5.7 Allocation of Duties and Functions
Prior to initiating a trial, the sponsor should define, establish, and allocate all trial-related duties and functions.

5.8 Compensation to Subjects and Investigators
5.8.1 If required by the applicable regulatory requirement(s), the sponsor should provide insurance or should indemnify the investigator/the institution against claims arising from the trial.

5.8.2 The sponsor's policies and procedures should address the costs of treatment of trial subjects in the event of trial-related injuries.

5.8.3 When trial subjects receive compensation, the method and manner of compensation should be in compliance with applicable regulatory requirement(s).

5.9 Financing
The financial aspects of the trial should be documented in an agreement between the sponsor and the investigator/institution.

5.10 Notification/Submission to Regulatory Authority(ies)
Before initiating the clinical trial(s), the sponsor (or the sponsor and the investigator, if required by the applicable regulatory requirement(s)) should submit any required application(s) to the appropriate authority(ies) for review, acceptance, and/or permission (as required by the applicable regulatory requirement(s)) to begin the trial(s).

5.11 Confirmation of Review by IRB/IEC
5.11.1 The sponsor should obtain from the investigator/institution documentation and dates of IRB/IEC approval.

5.11.2 If the IRB/IEC conditions its approval/favourable opinion upon change(s) in any aspect of the trial, such as modification(s) of the protocol, written informed consent form and any other written information to be provided to subjects, and/or other procedures, the sponsor should obtain from the investigator/institution a copy of the modification(s) made and the date approval/favourable opinion was given by the IRB/IEC.

5.11.3 The sponsor should obtain from the investigator/institution documentation and dates of any IRB/IEC re-approvals/re-evaluations with favourable opinion, and of any withdrawals or suspensions of approval/favourable opinion.

5.12 Information on Investigational Product(s)
5.12.1 When planning trials, the sponsor should ensure that sufficient safety and efficacy data from nonclinical studies and/or clinical trials are available to support human exposure by the route, at the dosages, for the duration, and in the trial population to be studied.

5.12.2 The sponsor should update the Investigator's Brochure as significant new information becomes available.

5.13 Manufacturing, Packaging, Labelling, and Coding Investigational Product(s)
5.13.1 The sponsor should ensure that the investigational product(s) (including active comparator(s) and placebo, if applicable) is characterized as appropriate to the stage of development of the product(s), is manufactured in accordance with any applicable GMP, and is coded and labelled in a manner that protects the blinding, if applicable.

5.13.2 The sponsor should determine, for the investigational product(s), acceptable storage temperatures, storage conditions, storage times, reconstitution fluids and procedures, and devices for product infusion, if any.

5.13.3 The investigational product(s) should be packaged to prevent contamination and unacceptable deterioration during transport and storage.

5.13.4 In blinded trials, the coding system for the investigational product(s) should include a mechanism that permits rapid identification of the product(s) in case of a medical emergency, but does not permit undetectable breaks of the blinding.

5.13.5 If significant formulation changes are made in the investigational or comparator product(s) during the course of clinical development, the results of any additional studies of the formulated product(s) (e.g., stability, dissolution rate, bioavailability) needed to assess whether these changes would significantly alter the pharmacokinetic profile of the product should be available prior to the use of the new formulation in clinical trials.

5.14 Supplying and Handling Investigational Product(s)
5.14.1 The sponsor is responsible for supplying the investigator(s)/institution(s) with the investigational product(s).

5.14.2 The sponsor should not supply an investigator/institution with the investigational product(s) until the sponsor obtains all required documentation.

5.14.3 The sponsor should ensure that written procedures include instructions that the investigator/institution should follow for the handling and storage of investigational product(s) for the trial and documentation thereof.

5.14.4 The sponsor should maintain sufficient quantities of the investigational product(s) used in the trials to reconfirm specifications, should the need arise, and maintain records of batch sample analyses and characteristics.

5.15 Record Access
5.15.1 The sponsor should ensure that it is specified in the protocol or other written agreement that the investigator(s)/institution(s) provide direct access to source data/documents for trial-related monitoring, audits, IRB/IEC review, and regulatory inspection.

5.15.2 The sponsor should verify that each subject has consented, in writing, to direct access to his/her original medical records for trial-related monitoring, audit, IRB/IEC review, and regulatory inspection.

5.16 Safety Information
5.16.1 The sponsor is responsible for the ongoing safety evaluation of the investigational product(s).

5.16.2 The sponsor should promptly notify all concerned investigator(s)/institution(s) and the regulatory authority(ies) of findings that could affect adversely the safety of subjects, impact the conduct of the trial, or alter the IRB's/IEC's approval/favourable opinion to continue the trial.

5.17 Adverse Drug Reaction Reporting
5.17.1 The sponsor should expedite the reporting to all concerned investigator(s)/institutions(s), to the IRB(s)/IEC(s), where required, and to the regulatory authority(ies) of all adverse drug reactions (ADRs) that are both serious and unexpected.

5.17.2 Such expedited reports should comply with the applicable regulatory requirement(s) and with the ICH Guideline for Clinical Safety Data Management: Definitions and Standards for Expedited Reporting.

5.17.3 The sponsor should submit to the regulatory authority(ies) all safety updates and periodic reports, as required by applicable regulatory requirement(s).

5.18 Monitoring
5.18.1 Purpose: The purposes of trial monitoring are to verify that the rights and well-being of human subjects are protected, that the reported trial data are accurate, complete, and verifiable from source documents, and that the conduct of the trial is in compliance with the currently approved protocol/amendment(s), with GCP, and with the applicable regulatory requirement(s).

5.18.2 Selection and Qualifications of Monitors: Monitors should be appointed by the sponsor and should be appropriately trained, and should have the scientific and/or clinical knowledge needed to monitor the trial adequately.

5.18.3 Extent of Monitoring: The sponsor should ensure that the trials are adequately monitored. The sponsor should determine the appropriate extent and nature of monitoring.

5.18.4 Monitor's Responsibilities: The monitor(s) in accordance with the sponsor's requirements should ensure that the trial is conducted and documented properly.

5.18.5 Monitoring Procedures: The monitor(s) should follow the sponsor's established written SOPs as well as those procedures that are specified by the sponsor for monitoring a specific trial.

5.18.6 Monitoring Report: The monitor should submit a written report to the sponsor after each trial-site visit or trial-related communication.

5.19 Audit
If or when sponsors perform audits, as part of implementing quality assurance, they should consider the purpose of an audit, the independence and qualifications of the auditors, procedures for auditing, reporting, and follow-up.

5.20 Noncompliance
5.20.1 Noncompliance with the protocol, SOPs, GCP, and/or applicable regulatory requirement(s) by an investigator/institution, or by member(s) of the sponsor's staff should lead to prompt action by the sponsor to secure compliance.

5.20.2 If the monitoring and/or auditing identifies serious and/or persistent noncompliance on the part of an investigator/institution, the sponsor should terminate the investigator's/institution's participation in the trial.

5.20.3 When an investigator's/institution's participation is terminated because of noncompliance, the sponsor should notify promptly the regulatory authority(ies).

5.21 Premature Termination or Suspension of a Trial
If a trial is prematurely terminated or suspended, the sponsor should promptly inform the investigators/institutions, and the regulatory authority(ies) of the termination or suspension and the reason(s) for the termination or suspension.

5.22 Clinical Trial/Study Reports
Whether the trial is completed or prematurely terminated, the sponsor should ensure that the clinical trial reports are prepared and provided to the regulatory agency(ies) as required by the applicable regulatory requirement(s).

5.23 Multicentre Trials
For multicentre trials, the sponsor should ensure that all investigators conduct the trial in strict compliance with the protocol agreed to by the sponsor and, if required, by the regulatory authority(ies), and given approval/favourable opinion by the IRB/IEC.
            """,
            "subsections": []
        },
        
        # Section 6: Clinical Trial Protocol
        {
            "section_id": "6",
            "title": "Clinical Trial Protocol and Protocol Amendment(s)",
            "content": """
The contents of a trial protocol should generally include the following topics. However, site specific information may be provided on separate protocol page(s), or addressed in a separate agreement, and some of the information listed below may be contained in other protocol referenced documents, such as an Investigator's Brochure.

6.1 General Information
6.1.1 Protocol title, protocol identifying number, and date. Any amendment(s) should also bear the amendment number(s) and date(s).

6.1.2 Name and address of the sponsor and monitor (if other than the sponsor).

6.1.3 Name and title of the person(s) authorized to sign the protocol and the protocol amendment(s) for the sponsor.

6.1.4 Name, title, address, and telephone number(s) of the sponsor's medical expert (or dentist when appropriate) for the trial.

6.1.5 Name and title of the investigator(s) who is (are) responsible for conducting the trial, and the address and telephone number(s) of the trial site(s).

6.1.6 Name, title, address, and telephone number(s) of the qualified physician (or dentist, if applicable), who is responsible for all trial-site related medical (or dental) decisions (if other than investigator).

6.1.7 Name(s) and address(es) of the clinical laboratory(ies) and other medical and/or technical department(s) and/or institutions involved in the trial.

6.2 Background Information
6.2.1 Name and description of the investigational product(s).

6.2.2 A summary of findings from nonclinical studies that potentially have clinical significance and from clinical trials that are relevant to the trial.

6.2.3 Summary of the known and potential risks and benefits, if any, to human subjects.

6.2.4 Description of and justification for the route of administration, dosage, dosage regimen, and treatment period(s).

6.2.5 A statement that the trial will be conducted in compliance with the protocol, GCP and the applicable regulatory requirement(s).

6.2.6 Description of the population to be studied.

6.2.7 References to literature and data that are relevant to the trial, and that provide background for the trial.

6.3 Trial Objectives and Purpose
A detailed description of the objectives and the purpose of the trial.

6.4 Trial Design
The scientific integrity of the trial and the credibility of the data from the trial depend substantially on the trial design.

6.4.1 A specific statement of the primary endpoints and the secondary endpoints, if any, to be measured during the trial.

6.4.2 A description of the type/design of trial to be conducted (e.g., double-blind, placebo-controlled, parallel design) and a schematic diagram of trial design, procedures and stages.

6.4.3 A description of the measures taken to minimize/avoid bias, including randomization and blinding.

6.4.4 A description of the trial treatment(s) and the dosage and dosage regimen of the investigational product(s).

6.4.5 The expected duration of subject participation, and a description of the sequence and duration of all trial periods, including follow-up, if any.

6.4.6 A description of the "stopping rules" or "discontinuation criteria" for individual subjects, parts of trial and entire trial.

6.4.7 Accountability procedures for the investigational product(s), including the placebo(s) and comparator(s), if any.

6.4.8 Maintenance of trial treatment randomization codes and procedures for breaking codes.

6.4.9 The identification of any data to be recorded directly on the CRFs (i.e., no prior written or electronic record of data), and to be considered to be source data.

6.5 Selection and Withdrawal of Subjects
6.5.1 Subject inclusion criteria.

6.5.2 Subject exclusion criteria.

6.5.3 Subject withdrawal criteria (i.e., terminating investigational product treatment/trial treatment) and procedures specifying when and how to withdraw subjects from the trial/investigational product treatment.

6.6 Treatment of Subjects
6.6.1 The treatment(s) to be administered, including the name(s) of all the product(s), the dose(s), the dosing schedule(s), the route/mode(s) of administration, and the treatment period(s).

6.6.2 Medication(s)/treatment(s) permitted (including rescue medication) and not permitted before and/or during the trial.

6.6.3 Procedures for monitoring subject compliance.

6.7 Assessment of Efficacy
6.7.1 Specification of the efficacy parameters.

6.7.2 Methods and timing for assessing, recording, and analysing of efficacy parameters.

6.8 Assessment of Safety
6.8.1 Specification of safety parameters.

6.8.2 The methods and timing for assessing, recording, and analysing safety parameters.

6.8.3 Procedures for eliciting reports of and for recording and reporting adverse event and intercurrent illnesses.

6.8.4 The type and duration of the follow-up of subjects after adverse events.

6.9 Statistics
6.9.1 A description of the statistical methods to be employed, including timing of any planned interim analysis(es).

6.9.2 The number of subjects planned to be enrolled. In multicentre trials, the numbers of enrolled subjects projected for each trial site should be specified.

6.9.3 The level of significance to be used.

6.9.4 Criteria for the termination of the trial.

6.9.5 Procedure for accounting for missing, unused, and spurious data.

6.9.6 Procedures for reporting any deviation(s) from the original statistical plan.

6.9.7 The selection of subjects to be included in the analyses.

6.10 Direct Access to Source Data/Documents
The sponsor should ensure that it is specified in the protocol or other written agreement that the investigator(s)/institution(s) will permit trial-related monitoring, audits, IRB/IEC review, and regulatory inspection(s), providing direct access to source data/documents.

6.11 Quality Control and Quality Assurance

6.12 Ethics
Description of ethical considerations relating to the trial.

6.13 Data Handling and Record Keeping

6.14 Financing and Insurance
Financing and insurance if not addressed in a separate agreement.

6.15 Publication Policy
Publication policy, if not addressed in a separate agreement.

6.16 Supplements
Protocol supplements are listed as follows:
- Protocol signature page
- Amendments
- Administrative pages
- Appendices and attachments
            """,
            "subsections": []
        },
        
        # Section 7: Investigator's Brochure
        {
            "section_id": "7",
            "title": "Investigator's Brochure",
            "content": """
7.1 Introduction
The Investigator's Brochure (IB) is a compilation of the clinical and nonclinical data on the investigational product(s) that are relevant to the study of the product(s) in human subjects.

Its purpose is to provide the investigators and others involved in the trial with the information to facilitate their understanding of the rationale for, and their compliance with, many key features of the protocol.

The IB also provides insight to support the clinical management of the study subjects during the course of the clinical trial.

7.2 General Considerations
7.2.1 The IB should include a summary of the relevant physical, chemical, pharmaceutical, pharmacological, toxicological, pharmacokinetic, metabolic, and clinical information available that is relevant to the stage of clinical development of the investigational product.

7.2.2 The IB should be revised as new and relevant information becomes available.

7.2.3 In general, the IB should present information in a brief, simple, objective, balanced, and non-promotional manner.

7.2.4 The IB should be reviewed and approved by the disciplines responsible for the data being presented.

7.2.5 A sponsor may provide the basic IB to investigators who are studying investigational products.

7.3 Contents of the Investigator's Brochure
The IB should contain the following sections, each with literature references where appropriate:

7.3.1 Title Page
7.3.2 Confidentiality Statement
7.3.3 Table of Contents
7.3.4 Summary
7.3.5 Introduction
7.3.6 Physical, Chemical, and Pharmaceutical Properties and Formulation
7.3.7 Nonclinical Studies
7.3.8 Effects in Humans
7.3.9 Summary of Data and Guidance for the Investigator
7.3.10 Appendices

7.4 Minimum Information Should Be Included
The IB should contain the minimum amount of information required for the particular stage of drug development.

7.5 Updates
The IB should be updated at least annually and reviewed by the sponsor.
            """,
            "subsections": []
        },
        
        # Section 8: Essential Documents
        {
            "section_id": "8",
            "title": "Essential Documents for the Conduct of a Clinical Trial",
            "content": """
8.1 Introduction
Essential Documents are those documents which individually and collectively permit evaluation of the conduct of a trial and the quality of the data produced. These documents serve to demonstrate the compliance of the investigator, sponsor and monitor with the standards of Good Clinical Practice and with all applicable regulatory requirements.

8.2 Before the Clinical Phase of the Trial Commences
During this planning stage the following documents should be generated and should be on file before the trial formally starts:

8.2.1 Investigator's Brochure
8.2.2 Signed Protocol and Amendments, and Sample CRF
8.2.3 Information Given to Trial Subject (Informed Consent Form, Written Information, Advertisement, etc.)
8.2.4 Financial Aspects of the Trial
8.2.5 Insurance Statement
8.2.6 Signed Agreement between Involved Parties
8.2.7 Dated, Documented Approval/Favorable Opinion of IRB/IEC
8.2.8 IRB/IEC Composition
8.2.9 Regulatory Authority Authorizations/Approvals
8.2.10 Curriculum Vitae and/or Other Relevant Documents
8.2.11 Normal Value(s)/Range(s) for Medical/Laboratory/Technical Procedures
8.2.12 Medical/Laboratory/Technical Procedures
8.2.13 Sample of Label(s) Attached to Investigational Product Container(s)
8.2.14 Instructions for Handling of Investigational Product(s) and Trial-Related Materials
8.2.15 Shipping Records for Investigational Product(s)
8.2.16 Certificate(s) of Analysis
8.2.17 Decoding Procedures for Blinded Trials
8.2.18 Master Randomization List
8.2.19 Pre-Trial Monitoring Report
8.2.20 Trial Initiation Monitoring Report

8.3 During the Clinical Conduct of the Trial
In addition to documents already listed, the following should be added to the files during the trial:

8.3.1 Investigator's Brochure Updates
8.3.2 Any Revision to Informed Consent Form, Written Information, Advertisement
8.3.3 Regulatory Authority Authorizations for Protocol Amendments
8.3.4 IRB/IEC Approval for Protocol Amendments
8.3.5 Curriculum Vitae for New Investigators and/or Sub-Investigators
8.3.6 Updates to Normal Value(s)/Range(s) for Medical/Laboratory/Technical Procedures
8.3.7 Updates of Medical/Laboratory/Technical Procedures
8.3.8 Documentation of Investigational Product(s) and Trial-Related Materials Shipment
8.3.9 Certificate(s) of Analysis for New Batches of Investigational Products
8.3.10 Monitoring Visit Reports
8.3.11 Relevant Communications (Letters, Meeting Notes, Notes of Telephone Calls)
8.3.12 Signed Informed Consent Forms
8.3.13 Source Documents
8.3.14 Signed, Dated, and Completed CRFs
8.3.15 Documentation of CRF Corrections
8.3.16 Notification by Originating Investigator to Sponsor of SAEs and Related Reports
8.3.17 Notification by Sponsor and/or Investigator to Regulatory Authorities and IRB(s)/IEC(s) of Unexpected Serious ADRs
8.3.18 Notification by Sponsor to Investigators of Safety Information
8.3.19 Interim or Annual Reports to IRB/IEC and Authorities
8.3.20 Subject Screening Log
8.3.21 Subject Identification Code List
8.3.22 Subject Enrollment Log
8.3.23 Investigational Products Accountability at the Site
8.3.24 Signature Sheet
8.3.25 Record of Retained Body Fluids/Tissue Samples

8.4 After Completion or Termination of the Trial
After completion or termination of the trial, all of the documents identified in sections 8.2 and 8.3 should be in the file together with the following:

8.4.1 Investigational Product(s) Accountability at Site
8.4.2 Documentation of Investigational Product Destruction
8.4.3 Completed Subject Identification Code List
8.4.4 Audit Certificate (if available)
8.4.5 Final Trial Close-Out Monitoring Report
8.4.6 Treatment Allocation and Decoding Documentation
8.4.7 Final Report by Investigator to IRB/IEC
8.4.8 Clinical Study Report
            """,
            "subsections": []
        }
    ]
}


def get_ich_gcp_guidelines() -> dict:
    """Return the complete ICH-GCP guidelines."""
    return ICH_GCP_GUIDELINES


def get_section(section_id: str) -> dict:
    """Get a specific section by ID."""
    for section in ICH_GCP_GUIDELINES["sections"]:
        if section["section_id"] == section_id:
            return section
    return None


def get_all_sections() -> list:
    """Get all sections."""
    return ICH_GCP_GUIDELINES["sections"]


def search_guidelines(query: str) -> list:
    """Simple text search in guidelines."""
    query_lower = query.lower()
    results = []
    for section in ICH_GCP_GUIDELINES["sections"]:
        if query_lower in section["content"].lower() or query_lower in section["title"].lower():
            results.append({
                "section_id": section["section_id"],
                "title": section["title"],
                "relevance": "high" if query_lower in section["title"].lower() else "medium"
            })
    return results
