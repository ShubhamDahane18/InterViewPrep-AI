from pydantic import BaseModel, EmailStr, HttpUrl, Field
from typing import Optional, List


class Project(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    github_link: Optional[HttpUrl] = Field(None, description="GitHub URL of the project")
    live_link: Optional[HttpUrl] = Field(None, description="Live demo or deployment link")
    time_period: Optional[str] = Field(None, description="Time duration or dates of the project")
    features: Optional[List[str]] = Field(default_factory=list, description="Key features of the project")
    tech_stack: Optional[List[str]] = Field(default_factory=list, description="Technologies used in the project")


class SkillsSection(BaseModel):
    programming_languages: Optional[List[str]] = Field(default_factory=list, description="Languages like Python, C++")
    frameworks: Optional[List[str]] = Field(default_factory=list, description="Frameworks like Django, React")
    libraries_tools: Optional[List[str]] = Field(default_factory=list, description="Libraries and tools like NumPy, Git")
    databases: Optional[List[str]] = Field(default_factory=list, description="Databases like MongoDB, MySQL")
    soft_skills: Optional[List[str]] = Field(default_factory=list, description="Soft skills like leadership, communication")


class ExtractResumeData(BaseModel):
    name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[EmailStr] = Field(None, description="Email address")
    mob_no: Optional[str] = Field(None, description="Mobile phone number")

    linkedin: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    github: Optional[HttpUrl] = Field(None, description="GitHub profile URL")
    leetcode: Optional[HttpUrl] = Field(None, description="LeetCode profile URL")

    skills: Optional[SkillsSection] = Field(default_factory=SkillsSection, description="Structured skill information")
    projects: Optional[List[Project]] = Field(default_factory=list, description="List of personal or academic projects")