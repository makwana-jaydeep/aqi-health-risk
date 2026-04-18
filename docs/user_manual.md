# User Manual

## AQI Health Risk Assessment Application

**For non-technical users**

---

## What This Application Does

This application tells you whether today's air quality in your city is safe for you
specifically, based on your age, health conditions, and what you plan to do outside.

Instead of just showing you a number, it gives you one of three clear recommendations:

- **Safe**: You can go outside normally today.
- **Caution**: You should limit time outdoors or wear protection.
- **Avoid Outdoors**: Stay indoors. The air is hazardous.

---

## How to Access the Application

Open your web browser and go to:
```
http://localhost:8501
```

---

## Step-by-Step: Getting Your Risk Assessment

### Step 1: Fill in Your Health Profile (left panel)

The left side panel asks three things:

**Age Group**
Click the dropdown and select the option that matches you:
- child (0-12)
- teen (13-17)
- adult (18-59)
- senior (60+)

**Health Conditions** (tick any that apply)
- I have asthma or respiratory conditions
- I have a heart condition

**Planned outdoor activity**
Select how active you plan to be outdoors today:
- low: sitting outside, slow walking
- moderate: brisk walking, cycling
- high: jogging, sports, heavy exercise

---

### Step 2: Enter Your City and Pollution Readings

**Select your city** from the dropdown list.

**Enter the current pollution readings** for your city. You can find these on:
- The CPCB SAMEER app
- Your city's pollution control board website
- The Airvisual or IQ Air app

Enter values for:
- **AQI**: The overall air quality index (use the slider)
- **PM2.5**: Fine particle concentration
- **PM10**: Coarse particle concentration
- **NO2**: Nitrogen dioxide level

---

### Step 3: Enter the Current Weather

Use the sliders to enter:
- **Temperature** in Celsius
- **Humidity** in percent
- **Wind Speed** in km/h

You can check weather on any weather app for your city.

---

### Step 4: Click "Assess My Risk"

Click the large blue button at the bottom of the page.

The application will show you your personal risk level in a coloured box:
- Green box = Safe
- Yellow box = Caution
- Red box = Avoid Outdoors

Below the box you will see a written recommendation explaining what to do.

---

## The Pipeline Monitor Page

Click "Pipeline Monitor" in the navigation to see technical information about the
system, including:
- Whether the AI model is loaded and ready
- When data was last updated
- Links to more detailed dashboards

You do not need to use this page for normal risk assessment. It is for system
administrators and data engineers.

---

## Frequently Asked Questions

**Q: Where does the AQI data come from?**
A: Live data comes from India's Central Pollution Control Board (CPCB) open data portal.

**Q: How accurate is the recommendation?**
A: The model is trained on thousands of data points and achieves over 85% accuracy.
   The recommendation is informational, not medical advice. Consult a doctor for
   any serious health concerns.

**Q: What should I do if the app shows an error?**
A: Refresh the page. If the error persists, contact your system administrator.

**Q: Does the app store my health information?**
A: No. The application does not store or log any personal health data.
   Your inputs are used only to compute the current recommendation and are
   discarded immediately.

---

## Contact and Support

For technical issues, contact your system administrator or the project team.
