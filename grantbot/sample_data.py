"""
Some shared data that are helpful during development
"""


RWF_SUMMARY = """
Ludlow-Taylor Elementary School, located in the Capitol Hill neighborhood of
Washington, DC, seeks funding to support its afterschool "Reading with Friends"
program for Kindergarten through 5th-grade students. This initiative provides
literacy support, particularly benefiting underserved and at-risk students who
have experienced significant learning loss due to the COVID-19 pandemic. The
program offers free participation for families with financial need, ensuring
equitable access to critical reading instruction. In the 2024-25 school year,
65 students per session will engage in small group reading activities aimed at
improving literacy skills and motivation. Funding will cover teacher salaries,
materials, and participation subsidies.
"""


RWF_PROGRAM_NAME = "Reading with Friends"


MAIN_INSTRUCTION = """
{program_summary}

Answer the following questions:

1. What is the grant name? (if applicable)
2. What is the grant link?

If it looks like the '{program_name}' program is not a good match
for this grantmaker, please answer the following question:

Why is the '{program_name}' program not a good match for this grantmaker? 

If it looks like a potentially good match, please also answer the following
questions:

1. What is the grant amount? This might be a range or a specific amount, or some other
   short description.
2. What are the eligibility criteria?
3. What's the application deadline? Oftentimes there are multiple deadlines throughout the
   year. Just briefly describe how this grantmaker organizes their deadlines.
4. Are there any additional notes or considerations for applicants?
5. What is the application procedure? Please comment specifically on
    whether the application can be submitted online.
6. What is the link to apply for the grant?
"""
