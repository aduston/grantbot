from pydantic import BaseModel


class GrantInformation(BaseModel):
    grant_maker: str
    grant_name: str | None
    grant_link: str
    amount: str
    eligibility: str
    deadline: str
    notes: str
    can_apply_online: bool
    application_procedure: str
