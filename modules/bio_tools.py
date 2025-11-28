# modules/bio_tools.py
"""
Enhanced health tools module.

Features:
- Flexible BMI parser: calculate_bmi_flexible(input_str or weight,height)
- TDEE calculator
- Local meal plan generator: generate_meal_plan(calorie_target, diet_type, days)
- Infermedica triage integration: infermedica_triage(symptoms_text, age, sex)
- get_health_tools() -> dict of callables for simple agent usage
- get_langchain_tools() -> list of LangChain Tool objects (if langchain installed)

Notes:
- This module avoids importing LangChain at module import time to prevent pulling heavy deps.
"""

from typing import Dict, Any, List, Callable, Optional
import os
import re
import requests
import math

# DO NOT import langchain at top-level (lazy import inside get_langchain_tools)

# ---------------------------
# BMI (flexible parser)
# ---------------------------

def _parse_weight_height_flexible(s: str):
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        w = float(s[0])
        h = float(s[1])
        if h > 3:
            h = h / 100.0
        return w, h

    text = str(s).lower().strip()
    nums = re.findall(r"\d+\.?\d*", text)
    kg_match = re.search(r"(\d+\.?\d*)\s*kg", text)
    cm_match = re.search(r"(\d+\.?\d*)\s*cm", text)
    m_match = re.search(r"(\d+\.?\d*)\s*m\b", text)

    if kg_match and (cm_match or m_match):
        w = float(kg_match.group(1))
        if cm_match:
            h = float(cm_match.group(1)) / 100.0
        else:
            h = float(m_match.group(1))
        return w, h

    if "=" in text or ";" in text:
        parts = re.split(r"[;,\s]+", text.replace("=", ";"))
        numeric = [p for p in parts if re.match(r"^\d+\.?\d*$", p)]
        if len(numeric) >= 2:
            w = float(numeric[0])
            h = float(numeric[1])
            if h > 3:
                h = h / 100.0
            return w, h

    if len(nums) >= 2:
        w = float(nums[0])
        h = float(nums[1])
        if h > 3:
            h = h / 100.0
        return w, h

    raise ValueError("Could not parse weight and height from input. Try formats like '75 1.8' or '75kg 180cm'.")


def calculate_bmi_flexible(input_val) -> Dict[str, Any]:
    try:
        if isinstance(input_val, (list, tuple)):
            w, h = _parse_weight_height_flexible(input_val)
        elif isinstance(input_val, (int, float)):
            raise ValueError("Single numeric input ambiguous; provide weight and height together.")
        else:
            w, h = _parse_weight_height_flexible(str(input_val))

        if h <= 0:
            raise ValueError("Height must be > 0 meters.")

        bmi = w / (h * h)
        bmi_rounded = round(bmi, 2)
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        text = f"BMI: {bmi_rounded}. Category: {category}."
        return {"bmi": bmi_rounded, "category": category, "text": text}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# TDEE (kept, improved)
# ---------------------------

def calculate_tdee(weight_kg: float, height_cm: float, age: int, gender: str = "male", activity_level: str = "sedentary") -> Dict[str, Any]:
    try:
        w = float(weight_kg)
        h = float(height_cm)
        a = int(age)
        g = str(gender).lower()
        if g not in ("male", "female"):
            g = "male"

        if g == "male":
            bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
        else:
            bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

        multipliers = {'sedentary': 1.2, 'moderate': 1.55, 'active': 1.725}
        multiplier = multipliers.get(activity_level.lower(), 1.2)
        tdee = int(round(bmr * multiplier))
        return {"tdee": tdee, "bmr": round(bmr, 1), "text": f"Estimated daily calories to maintain weight: {tdee} kcal (BMR ≈ {int(round(bmr))} kcal)."}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# Meal plan generator
# ---------------------------

_DEFAULT_MEALS = {
    "balanced": {
        "breakfast": ["Oatmeal with fruit", "2 boiled eggs"],
        "lunch": ["Grilled chicken salad", "Quinoa or brown rice"],
        "dinner": ["Baked salmon, steamed veggies", "Sweet potato"],
        "snack": ["Greek yogurt", "Handful of nuts"]
    },
    "keto": {
        "breakfast": ["Scrambled eggs with avocado", "Cheese"],
        "lunch": ["Grilled salmon salad with olive oil", "Olives"],
        "dinner": ["Steak with buttered broccoli", "Cauliflower mash"],
        "snack": ["Almonds", "Cheese slices"]
    },
    "vegan": {
        "breakfast": ["Smoothie with plant protein", "Oats"],
        "lunch": ["Lentil salad with veggies", "Tofu"],
        "dinner": ["Chickpea curry with brown rice", "Mixed veg"],
        "snack": ["Fruit", "Hummus and carrots"]
    },
    "paleo": {
        "breakfast": ["Eggs and spinach", "Berries"],
        "lunch": ["Grilled chicken and veggies", "Sweet potato"],
        "dinner": ["Roasted fish and salad", "Steamed greens"],
        "snack": ["Mixed nuts", "Beef jerky (minimal processing)"]
    }
}

def _pick_meal(meal_list: List[str], day_index: int) -> str:
    return meal_list[day_index % len(meal_list)]

def generate_meal_plan(calorie_target: int = 2000, diet_type: str = "balanced", days: int = 7) -> Dict[str, Any]:
    try:
        diet = diet_type.lower()
        if diet not in _DEFAULT_MEALS:
            diet = "balanced"
        plan = {}
        per_meal = {
            "breakfast": 0.25,
            "lunch": 0.35,
            "dinner": 0.30,
            "snack": 0.10
        }
        for d in range(days):
            day_name = f"Day {d+1}"
            breakfast = _pick_meal(_DEFAULT_MEALS[diet]["breakfast"], d)
            lunch = _pick_meal(_DEFAULT_MEALS[diet]["lunch"], d)
            dinner = _pick_meal(_DEFAULT_MEALS[diet]["dinner"], d)
            snack = _pick_meal(_DEFAULT_MEALS[diet]["snack"], d)
            b_kcal = int(calorie_target * per_meal["breakfast"])
            l_kcal = int(calorie_target * per_meal["lunch"])
            d_kcal = int(calorie_target * per_meal["dinner"])
            s_kcal = int(calorie_target * per_meal["snack"])
            plan[day_name] = {
                "breakfast": {"item": breakfast, "approx_kcal": b_kcal},
                "lunch": {"item": lunch, "approx_kcal": l_kcal},
                "dinner": {"item": dinner, "approx_kcal": d_kcal},
                "snack": {"item": snack, "approx_kcal": s_kcal}
            }
        summary = f"{days}-day {diet.capitalize()} meal plan (~{calorie_target} kcal/day)."
        return {"calorie_target": calorie_target, "diet_type": diet, "days": days, "plan": plan, "text": summary}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# Infermedica triage integration
# ---------------------------

_INFERMEDICA_PARSE_URL = "https://api.infermedica.com/v3/parse"
_INFERMEDICA_DIAGNOSE_URL = "https://api.infermedica.com/v3/diagnosis"

def _get_infermedica_headers(app_id: Optional[str], app_key: Optional[str]) -> Dict[str, str]:
    aid = app_id or os.environ.get("INFERMEDICA_APP_ID")
    akey = app_key or os.environ.get("INFERMEDICA_APP_KEY")
    if not aid or not akey:
        return {}
    return {
        "App-Id": aid,
        "App-Key": akey,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def infermedica_triage(symptoms_text: str, age: int = 30, sex: str = "male", app_id: Optional[str] = None, app_key: Optional[str] = None) -> Dict[str, Any]:
    headers = _get_infermedica_headers(app_id, app_key)
    if not headers:
        return {"error": "Infermedica keys not configured. Set INFERMEDICA_APP_ID and INFERMEDICA_APP_KEY."}

    try:
        parse_payload = {"text": symptoms_text}
        r = requests.post(_INFERMEDICA_PARSE_URL, json=parse_payload, headers=headers, timeout=10)
        if r.status_code != 200:
            return {"error": f"Infermedica /parse failed with status {r.status_code}: {r.text}"}
        parse_resp = r.json()
        mentions = parse_resp.get("mentions", [])
        evidence = []
        for m in mentions:
            mid = m.get("id")
            choice = m.get("choice_id", "present")
            if mid:
                evidence.append({"id": mid, "choice_id": choice, "source": "user"})
        if not evidence:
            return {"error": "Could not map symptoms to known clinical terms. Please provide concise symptoms (e.g., 'fever and sore throat for 2 days')."}
    except Exception as e:
        return {"error": f"Error calling Infermedica /parse: {e}"}

    try:
        diagnose_payload = {
            "sex": sex.lower(),
            "age": {"value": int(age)},
            "evidence": evidence
        }
        r2 = requests.post(_INFERMEDICA_DIAGNOSE_URL, json=diagnose_payload, headers=headers, timeout=12)
        if r2.status_code != 200:
            return {"error": f"Infermedica /diagnosis failed with status {r2.status_code}: {r2.text}"}
        diag = r2.json()
        triage_text = "Infermedica diagnosis returned. Probable conditions & triage suggestions included."
        return {"triage": diag, "text": triage_text}
    except Exception as e:
        return {"error": f"Error calling Infermedica /diagnosis: {e}"}


# ---------------------------
# Tools registry (simple agent)
# ---------------------------

def get_health_tools() -> Dict[str, Callable]:
    return {
        "calculate_bmi": calculate_bmi_flexible,
        "calculate_tdee": calculate_tdee,
        "generate_meal_plan": generate_meal_plan,
        "infermedica_triage": infermedica_triage,
        "web_search": lambda q: "Web search not configured in this toolset. Use langchain or Tavily if available."
    }


# ---------------------------
# LangChain tool wrappers (optional, lazy)
# ---------------------------

def get_langchain_tools() -> List[Any]:
    """
    Try to lazily import and return LangChain Tool objects.
    Returns [] if langchain is not installed or creation fails.
    """
    try:
        from langchain.tools import Tool as LC_Tool  # lazy import
    except Exception:
        return []

    tools = []
    try:
        tools.append(LC_Tool(name="calculate_bmi", func=lambda s: _lc_wrap_bmi(s), description="Compute BMI from flexible input (e.g. '75 1.8')."))
        tools.append(LC_Tool(name="calculate_tdee", func=lambda s: _lc_wrap_tdee(s), description="Compute TDEE. Input: 'weight;height;age;gender;activity'"))
        tools.append(LC_Tool(name="generate_meal_plan", func=lambda s: _lc_wrap_mealplan(s), description="Generate meal plan. Input: 'calories;diet;days' e.g. '2000;keto;7'"))
        tools.append(LC_Tool(name="infermedica_triage", func=lambda s: _lc_wrap_infermedica(s), description="Run Infermedica triage. Input: 'age|sex|symptoms' e.g. '32|male|fever and cough'"))
    except Exception:
        return []
    return tools


# ---------------------------
# LangChain wrapper helpers (string parsing)
# ---------------------------

def _lc_wrap_bmi(s: str) -> str:
    try:
        res = calculate_bmi_flexible(s)
        if "error" in res:
            return res["error"]
        return res["text"]
    except Exception as e:
        return f"Error: {e}"

def _lc_wrap_tdee(s: str) -> str:
    try:
        parts = [p.strip() for p in re.split(r"[;,|]", s) if p.strip()]
        if len(parts) < 5:
            return "TDEE input needs 5 values: weight_kg;height_cm;age;gender;activity"
        w, h, age, gender, activity = parts[0], parts[1], parts[2], parts[3], parts[4]
        res = calculate_tdee(float(w), float(h), int(age), gender, activity)
        return res.get("text", str(res))
    except Exception as e:
        return f"Error parsing TDEE input: {e}"

def _lc_wrap_mealplan(s: str) -> str:
    try:
        parts = [p.strip() for p in re.split(r"[;,|]", s) if p.strip()]
        if len(parts) < 3:
            return "Meal plan input needs 3 values: calories;diet;days e.g. '2000;keto;7'"
        calories = int(float(parts[0]))
        diet = parts[1]
        days = int(parts[2])
        res = generate_meal_plan(calories, diet, days)
        return res.get("text", str(res))
    except Exception as e:
        return f"Error parsing meal plan input: {e}"

def _lc_wrap_infermedica(s: str) -> str:
    try:
        parts = s.split("|", 2)
        if len(parts) < 3:
            return "Infermedica input format: 'age|sex|symptoms' e.g. '32|male|fever and cough'"
        age = int(parts[0])
        sex = parts[1]
        symptoms = parts[2]
        res = infermedica_triage(symptoms, age, sex)
        if "error" in res:
            return res["error"]
        return "Infermedica triage completed."
    except Exception as e:
        return f"Error parsing infermedica input: {e}"