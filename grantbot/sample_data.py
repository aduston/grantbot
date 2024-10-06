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

We understand that {grant_maker} has funded similar programs.
Please research {grant_maker} and provide the following  information:

1. Grant Name (if applicable)
2. Grant Link

If it looks like the '{program_name}' program is not a good match
for this grantmaker, please provide a brief explanation. If it
looks like a potentially good match, please also provide the
following information:

1. Grant Amount. This might be a range or a specific amount, or some other
   short description.
2. Eligibility Criteria
3. Application Deadline. Oftentimes there are multiple deadlines throughout the
   year. Just briefly describe how this grantmaker organizes their deadlines.
4. Any additional notes or considerations for applicants.
5. The application procedure. Please comment specifically on
    whether the application can be submitted online.
"""
