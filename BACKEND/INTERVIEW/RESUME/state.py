from pydantic import BaseModel, HttpUrl, EmailStr
from typing import Optional, List


class Project(BaseModel):
    project_name: str = ''
    github_link: Optional[HttpUrl] = ''
    live_link: Optional[HttpUrl] = ''
    time_period: Optional[str] = ''
    features: Optional[List[str]] = []
    tech_stack: Optional[List[str]] = []


class SkillsSection(BaseModel):
    programming_languages: Optional[List[str]] = []
    frameworks: Optional[List[str]] = []
    libraries_tools: Optional[List[str]] = []
    databases: Optional[List[str]] = []
    soft_skills: Optional[List[str]] = []


class ResumeAgentState(BaseModel):
    message : Optional[str] = ''
    full_text: Optional[str] = ''
    links: Optional[List[dict]] = []

    # Personal Info
    name: Optional[str] = ''
    email: Optional[EmailStr] = ''
    mob_no: Optional[str] = ''

    # Profile Links
    linkedin: Optional[HttpUrl] = ''
    github: Optional[HttpUrl] = ''
    leetcode: Optional[HttpUrl] = ''

    # Skills and Projects
    skills: Optional[SkillsSection] = SkillsSection()
    projects: Optional[List[Project]] = []