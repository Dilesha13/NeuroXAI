Design a professional medical SaaS web application UI for a system called NeuroXAI – AI-Powered Neonatal Seizure Detection Platform.

This platform is used by neurologists, pediatricians, and hospital staff to analyze EEG recordings from newborn babies and detect seizures using artificial intelligence. The interface should look like modern hospital software or healthcare analytics dashboards, similar to professional medical SaaS products used in real clinical environments.

The design must be clean, structured, trustworthy, modern, and highly usable for healthcare professionals.

Core Requirements

Build a full responsive web UI

Include both Light Theme and Dark Theme

Use Tailwind CSS only for styling

Do not use any other styling libraries

Do not use custom CSS frameworks besides Tailwind

Use a clean component-based design system

Make the interface suitable for desktop, tablet, and mobile

Prioritize usability, readability, and accessibility

Visual Style

Use a professional healthcare SaaS design language.

Light Theme

Background: white / very light gray

Primary: deep medical blue

Secondary: teal

Accent: soft purple

Success: soft green

Warning: amber

Error: muted red

Dark Theme

Background: deep navy / charcoal

Cards: slightly lighter dark surfaces

Text: soft white / light gray

Accents: teal, blue, and purple glow highlights

Maintain strong contrast and readability

Dark mode should feel premium, clinical, and modern, not gaming-like

General Design Style

clean layout

rounded cards

subtle shadows

structured dashboard layout

clear typography hierarchy

modern sans-serif typography such as Inter / Roboto style

lots of whitespace

clear status colors for medical interpretation

polished medical SaaS appearance

Responsiveness Requirements

The design must be fully responsive:

Desktop

full sidebar + top navbar

multiple dashboard cards in rows

large analytical charts

tables with filters and actions

Tablet

collapsible sidebar

grid layouts adjust to 2 columns

optimized spacing and card stacking

Mobile

sidebar becomes drawer / hamburger menu

forms become single-column

charts and cards stack vertically

navigation remains easy and user-friendly

no broken layouts or overflowing tables

Design should clearly show how components adapt across screen sizes.

Styling Constraint

Use Tailwind CSS only for all styling.

Important:

Use only Tailwind utility classes for visual styling

Avoid other styling systems

Avoid CSS-in-JS styling libraries

Avoid Bootstrap, Material UI styling systems, Chakra, Mantine, Ant, etc.

The final design should look polished using Tailwind alone

Medical Visual Elements

Include tasteful and gentle medical imagery / illustration areas such as:

newborn baby care visuals

EEG brain signal visualization

neural activity / brain graphics

doctors analyzing EEG data

neonatal healthcare illustrations

brain heatmap style visuals

These should feel:

gentle

caring

scientific

trustworthy

Not dramatic, scary, or overly technical.

Pages to Generate

Create a complete SaaS product UI with these screens.

1. Landing Page

Hero section:

Title: AI-Powered Neonatal Seizure Detection

Subtitle: Explain that NeuroXAI helps doctors detect seizures in newborns early using advanced EEG analysis and explainable artificial intelligence.

Hero visual: a baby + brain EEG signal visualization illustration

Buttons:

Get Started

Sign In

Sections:

How it Works

Upload EEG Recording

AI Analysis

Seizure Detection with Explainable Insights

Benefits

Early seizure detection

AI-assisted diagnosis

Explainable brain signal analysis

Clinical decision support

Add soft neonatal healthcare illustrations between sections.

Also show:

top navigation

theme toggle (light/dark)

responsive hero layout

2. Authentication Pages
Sign In Page

Centered login card with:

Title: Sign in to NeuroXAI

Email field

Password field

Remember me

Forgot password

Sign In button

Link: Don't have an account? Sign Up

Background:

soft gradient

subtle EEG or neonatal illustration

light and dark theme versions

Sign Up Page

Title: Create NeuroXAI Account

Full Name

Email

Password

Confirm Password

Hospital / Organization

Create Account button

Link: Already have an account? Sign In

Include a subtle baby + brain illustration.
Must be responsive and clean in both themes.

3. Main Dashboard

Create a professional medical dashboard after login.

Layout:

left sidebar navigation

top navigation bar

main analytics content area

Sidebar Navigation

Include:

Dashboard

Upload EEG

Patient Records

AI Analysis

Seizure Detection Results

Explainability

Reports

Settings

Logout

Add a theme toggle in header or profile area.

Dashboard Overview

Show summary cards:

Total EEG analyses

Seizure detections

Normal recordings

Active patients

Charts:

EEG activity statistics

Seizure detection trends

Model confidence distribution

Include:

recent patient analysis table

brain waveform visualization panel

activity timeline

status badges

Must look like real healthcare dashboard software.

4. Upload EEG Page

Title:
Upload EEG Recording

Components:

drag-and-drop upload area

supported format: EDF

patient ID field

recording date

upload progress bar

validation states

upload action button

Add EEG waveform visual beside or below upload section.

Responsive behavior:

side-by-side on desktop

stacked on mobile

5. Patient Records Page

Create a structured medical records table.

Columns:

Patient ID

Recording date

Duration

Status

Analysis result

Actions

Filters:

date

seizure detected / normal

patient search

sort options

Include pagination and search bar.

Responsive behavior:

desktop table

mobile cards or stacked rows

6. AI Analysis Results Page

Display detailed output after model analysis.

Sections:

Patient Information Card

Patient ID

Recording duration

Analysis timestamp

Prediction Result Card

Seizure Detected / No Seizure

Probability score

Confidence level

status color indicator

Visual Panels

EEG waveform visualization

seizure detection timeline

summary metrics

Use:

green for normal

red for seizure detected

maintain professional clinical tone

7. Explainable AI Page

Show explainability visualizations.

Panels:

EEG saliency heatmap

graph attention visualization

temporal attention map

Include descriptive helper text:
“Highlighted regions indicate EEG signal segments that most influenced the AI model's prediction.”

Also include:

explanation cards

top important EEG channels

clinical note section

Must feel understandable and trustworthy.

8. Reports Page

Generate polished report UI.

Components:

patient analysis report card

detection summary

downloadable report action

PDF export button

Visuals:

seizure probability trend

EEG overview snapshot

summary chart cards

9. Settings Page

Sections:

User Profile

name

email

hospital

profile image area

Security

change password

session preferences

App Settings

notification preferences

theme preference

export preferences

UI Components to Include

Use reusable Tailwind-styled components such as:

cards

tables

charts

tabs

dropdown filters

upload boxes

progress bars

status badges

modals

responsive navigation

theme toggle

Dark Theme Requirement

Generate the interface so that it clearly supports both light mode and dark mode.

Dark mode should include:

dark sidebar

dark cards

dark top navbar

readable charts and tables

clear status colors

polished premium appearance

The light and dark themes must feel like the same product system.

Final Goal

Generate a complete responsive medical SaaS product UI for AI-powered neonatal seizure detection, with:

light theme

dark theme

responsive layouts

Tailwind-only styling

professional hospital software appearance

neonatal + brain imagery

sign in / sign up

dashboard + records + reports + explainability pages

The final design should feel:

clinically trustworthy

polished

modern

data-focused

attractive for a final year project demo