# src/schemas.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

SkillLevel = Literal["beginner", "intermediate", "advanced", "expert", "unknown"]


class ProjectItem(BaseModel):
    """
    프로젝트 경험 1개.
    - 전부 채우게 강요하지 않음.
    - title/summary/tech_stack/domain 중 하나라도 있으면 유효.
    """
    title: Optional[str] = Field(default=None, description="프로젝트 이름/주제")
    summary: Optional[str] = Field(default=None, description="프로젝트 한줄 요약")
    tech_stack: List[str] = Field(default_factory=list, description="사용 기술/도구")
    domain: Optional[str] = Field(default=None, description="도메인(제조/금융/헬스 등)")

    def is_meaningful(self) -> bool:
        t = (self.title or "").strip()
        s = (self.summary or "").strip()
        d = (self.domain or "").strip()
        tech = any((x or "").strip() for x in self.tech_stack)
        return bool(t or s or d or tech)


class LanguageSkill(BaseModel):
    """언어/기술 + 숙련도 (evidence 없음)"""
    name: str = Field(..., description="언어/기술 이름 (예: Python, SQL, Airflow)")
    level: SkillLevel = Field(default="unknown", description="숙련도")


class UserProfile(BaseModel):
    # 6 슬롯 (누적 저장)
    project_experience: List[ProjectItem] = Field(default_factory=list)
    project_role: List[str] = Field(default_factory=list)
    languages: List[LanguageSkill] = Field(default_factory=list)
    preferred_work: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    major: Optional[str] = Field(default=None)


class ProfileUpdate(BaseModel):
    """이번 턴에서 새로 추출한 것만(부분 업데이트). None이면 업데이트 없음."""
    project_experience: Optional[List[ProjectItem]] = Field(default=None)
    project_role: Optional[List[str]] = Field(default=None)
    languages: Optional[List[LanguageSkill]] = Field(default=None)
    preferred_work: Optional[List[str]] = Field(default=None)
    interests: Optional[List[str]] = Field(default=None)
    major: Optional[str] = Field(default=None)


class ExtractionResult(BaseModel):
    """
    추출 전용 결과 (PydanticOutputParser로 파싱)
    - 사용자 메시지에서 명시된 내용만 profile_update에 담는다.
    """
    profile_update: ProfileUpdate = Field(default_factory=ProfileUpdate)


# -------------------------
# merge + filled 계산
# -------------------------

def _dedup_str_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def merge_profile(profile: UserProfile, update: ProfileUpdate) -> UserProfile:
    """기존 profile에 update를 '추가/보강' 방식으로 합침."""
    p = profile.model_copy(deep=True)

    if update.project_experience is not None:
        meaningful = [it for it in update.project_experience if it and it.is_meaningful()]
        if meaningful:
            existing = set()
            for it in p.project_experience:
                key = (
                    (it.title or "").strip().lower(),
                    (it.summary or "").strip().lower(),
                    ",".join(sorted([x.strip().lower() for x in it.tech_stack if (x or "").strip()])),
                )
                existing.add(key)

            for it in meaningful:
                key = (
                    (it.title or "").strip().lower(),
                    (it.summary or "").strip().lower(),
                    ",".join(sorted([x.strip().lower() for x in it.tech_stack if (x or "").strip()])),
                )
                if key != ("", "", "") and key in existing:
                    continue
                existing.add(key)
                p.project_experience.append(it)

    if update.project_role is not None and update.project_role:
        p.project_role = _dedup_str_list(p.project_role + update.project_role)

    if update.languages is not None and update.languages:
        order = {"unknown": 0, "beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        by_name = {ls.name.strip().lower(): ls for ls in p.languages if (ls.name or "").strip()}
        for ls in update.languages:
            k = (ls.name or "").strip().lower()
            if not k:
                continue
            if k not in by_name:
                by_name[k] = ls
            else:
                cur = by_name[k]
                if order.get(ls.level, 0) > order.get(cur.level, 0):
                    cur.level = ls.level
        p.languages = list(by_name.values())

    if update.preferred_work is not None and update.preferred_work:
        p.preferred_work = _dedup_str_list(p.preferred_work + update.preferred_work)

    if update.interests is not None and update.interests:
        p.interests = _dedup_str_list(p.interests + update.interests)

    if update.major is not None and update.major:
        if not p.major:
            p.major = update.major.strip()

    return p


def compute_filled_and_missing(profile: UserProfile) -> tuple[int, List[str]]:
    missing = []
    has_project = any(it.is_meaningful() for it in profile.project_experience)
    if not has_project:
        missing.append("project_experience")
    if not profile.project_role:
        missing.append("project_role")
    if not profile.languages:
        missing.append("languages")
    if not profile.preferred_work:
        missing.append("preferred_work")
    if not profile.interests:
        missing.append("interests")
    if not profile.major:
        missing.append("major")

    return 6 - len(missing), missing
