const express = require("express");
const axios = require("axios");
const { NlpManager } = require("node-nlp");
const cookieParser = require("cookie-parser");
const { v4: uuidv4 } = require("uuid");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const app = express();
app.use(express.json());
app.use(cookieParser());
const natural = require("natural");


const spellcheck = new natural.Spellcheck([
  "rwanda",
  "bulbula",
  "branch",
  "location",
  "address",
  "doctor",
  "specialty",
  "emergency",
  "contact",
  "services",
  "departments",
  "availability",
  "day",  
  "doctors",
  "specialist",
  "appointment",
  "outpatient",
  
]);

const branches = {
  "Rwanda Branch": [
    "Internal Medicine",
    "Obstetrics and Gynecology",
    "Pediatrics",
    "Pulmonology",
    "Gastroenterology",
    "Nephrology",
    "Endocrinology",
    "Neurology",
    "Oncology",
    "Cardiology",
    "Rheumatology",
    "ENT",
    "Hematology",
    "Psychiatry",
    "Dermatology",
    "Dialysis",
    "Endoscopy",
    "Spirometry",
    "Adult ICU",
    "Out Patient Department",
    "Emergency Services",
    "Imaging Services",
    "Laboratory Services",
    "NICU",
    "Advanced Life Support Ambulance Services",
    "Travel Medicine",
  ],
  "Bulbula Branch": [
    "General Surgery",
    "Orthopedic Surgery",
    "Laparoscopic Surgery",
    "Endocrine Surgery",
    "Plastic Surgery",
    "Vascular Surgery",
    "ENT Surgery",
    "Neurosurgery",
    "Hip and Knee Replacement Surgery",
    "Uro-surgery",
    "Hepatobiliary surgery",
    "Colorectal Surgery",
    "Pediatrics Surgery",
    "Internal Medicine",
    "Obstetrics and Gynecology",
    "Pediatrics",
    "Out Patient Department",
    "Emergency Services",
    "Adult ICU",
    "Laboratory Services",
    "Advanced Life Support Ambulance Services",
  ],
};

// Branch locations
const branchLocations = {
  "Rwanda Branch": "in front of Rwanda Embassy",
  "Bulbula Branch": "Bole Bulbula, around Mariam Mazoriya"
};

const contacts = ["6511", "+251-939515151", "+251-939525252"];
const days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const manager = new NlpManager({ languages: ["en"], nlu: { log: false } });

let cachedDoctors = [];
const sessionMemory = {}; // Multi-turn session memory

const setupNLP = async () => {
  const { data: doctors } = await axios.get(
    "https://appointment.whethiopia.com/filter/getAppointmentOpdUnitList.php"
  );
  cachedDoctors = doctors;

  // Named entities
  days.forEach((day) => manager.addNamedEntityText("day", day, ["en"], [day]));
  Object.keys(branchLocations).forEach(branchName => {
    manager.addNamedEntityText("branch", branchName, ["en"], [branchName]);
  });

  const doctorNames = [...new Set(doctors.map((d) => d.doctorName.split(" (")[0]))];
  doctorNames.forEach((name) =>
    manager.addNamedEntityText("doctor", name, ["en"], [name])
  );

  const specialties = [
    ...new Set(
      doctors.map((d) => d.specialityTitle?.toLowerCase()).filter(Boolean)
    ),
  ];
  specialties.forEach((spec) =>
    manager.addNamedEntityText("specialty", spec, ["en"], [spec])
  );

  // Intents
  manager.addDocument("en", "when is %doctor% available", "doctor.availability");
  manager.addDocument("en", "availability of %doctor%", "doctor.availability");
  manager.addDocument("en", "what day can I visit %doctor%", "doctor.availability");
  manager.addDocument("en", "doctor working on %day%", "doctor.by_day");
  manager.addDocument("en", "which doctors are available on %day%", "doctor.by_day");
  manager.addDocument("en", "show doctors for %specialty%", "doctor.by_specialty");
  manager.addDocument("en", "who is the %specialty%", "doctor.by_specialty");
  manager.addDocument("en", "i want to see a %specialty%", "doctor.by_specialty");

  manager.addDocument("en", "which branches do you have", "list.branches");
  manager.addDocument("en", "what branches do you have", "list.branches");

  manager.addDocument("en", "services in rwanda", "services.rwanda");
  manager.addDocument("en", "departments in bulbula branch", "services.bulbula");

  manager.addDocument("en", "list doctors", "list.doctors");

  manager.addDocument("en", "emergency number", "emergency.contact");
  manager.addDocument("en", "how do I contact in emergency", "emergency.contact");

  // Location intent
  manager.addDocument("en", "where is %branch%", "branch.location");
  manager.addDocument("en", "location of %branch%", "branch.location");
  manager.addDocument("en", "how can I find %branch%", "branch.location");
  manager.addDocument("en", "address of %branch%", "branch.location");
  manager.addDocument("en", "branch %branch% location", "branch.location");

  await manager.train();
  await manager.save();
};

setupNLP();

// Pronoun resolution
function resolvePronouns(input, memory) {
  let output = input;

  if (memory.branch) {
    output = output.replace(/\b(it|that branch|this branch|branch)\b/gi, memory.branch);
  }
  if (memory.doctors && memory.doctors.length > 1) {
    output = output.replace(/\b(they|them|those doctors|these doctors|doctors)\b/gi, memory.doctors.join(", "));
  }
  if (memory.doctor) {
    output = output.replace(/\b(him|her|that doctor|he|she|this doctor|doctor)\b/gi, memory.doctor);
  }
  if (memory.specialty) {
    output = output.replace(/\b(that specialty|this specialty|specialist|specialty)\b/gi, memory.specialty);
  }
  if (memory.day) {
    output = output.replace(/\b(that day|this day|day)\b/gi, memory.day);
  }

  return output;
}

// Main search route
app.post("/search", async (req, res) => {
  let sessionId = req.cookies.sessionId;
  if (!sessionId) {
    sessionId = uuidv4();
    res.cookie("sessionId", sessionId, {
      httpOnly: true,
      maxAge: 5 * 60 * 1000
    });
  }

  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });
const correctedQuestion = question
  .split(" ")
  .map((word) => {
    const correction = spellcheck.getCorrections(word.toLowerCase(), 1)[0];
    return correction || word;
  })
  .join(" ");

  if (!sessionMemory[sessionId]) sessionMemory[sessionId] = {};
  const memory = sessionMemory[sessionId];

  const resolvedQuestion = resolvePronouns(correctedQuestion, memory);
  const result = await manager.process("en", resolvedQuestion);
  const { intent, entities } = result;

  entities.forEach((e) => {
    memory[e.entity] = e.sourceText;
  });

  // Branch location
  if (intent === "branch.location") {
    const branchEntity =
      entities.find((e) => e.entity === "branch")?.sourceText || memory.branch;

    if (!branchEntity) {
      return res.json({ answer: "Please specify a branch name." });
    }

    const normalizedBranch = Object.keys(branchLocations).find(
      (b) => b.toLowerCase() === branchEntity.toLowerCase()
    );

    if (!normalizedBranch) {
      return res.json({ answer: `Sorry, I don't have the location for "${branchEntity}".` });
    }

    memory.branch = normalizedBranch;

    return res.json({
      answer: `Our ${normalizedBranch} is located ${branchLocations[normalizedBranch]}.`
    });
  }

  // List branches
  if (intent === "list.branches") {
    return res.json({ answer: Object.keys(branches) });
  }

  if (intent === "services.rwanda") {
    return res.json({ answer: branches["Rwanda Branch"] });
  }

  if (intent === "services.bulbula") {
    return res.json({ answer: branches["Bulbula Branch"] });
  }

  if (intent === "emergency.contact") {
    return res.json({ answer: `Call: ${contacts.join(", ")}` });
  }

  if (intent === "list.doctors") {
    return res.json({ answer: cachedDoctors.map((d) => d.doctorName) });
  }

  if (intent === "doctor.availability") {
    let doctorNames = [];
    if (memory.doctors && memory.doctors.length > 1) {
      doctorNames = memory.doctors;
    } else if (entities.find((e) => e.entity === "doctor")) {
      doctorNames = [entities.find((e) => e.entity === "doctor").sourceText];
    } else if (memory.doctor) {
      doctorNames = [memory.doctor];
    }
    if (doctorNames.length === 0) return res.json({ answer: "Please provide a doctor's name." });

    const answers = [];
    for (const name of doctorNames) {
      const doctor = cachedDoctors.find((d) =>
        d.doctorName.toLowerCase().includes(name.toLowerCase())
      );
      if (!doctor) {
        answers.push(`Doctor "${name}" not found.`);
        continue;
      }
      const availability = days
        .filter((day) => doctor[day] === "1")
        .map((day) => {
          const start = doctor[`availableStartTime${capitalize(day)}`];
          const end = doctor[`availableEndTime${capitalize(day)}`];
          return `${capitalize(day)}: ${start} - ${end}`;
        });
      answers.push(`${doctor.doctorName} is available on:`, ...availability, "");
    }
    return res.json({ answer: answers });
  }

  if (intent === "doctor.by_day") {
    const day =
      entities.find((e) => e.entity === "day")?.sourceText?.toLowerCase() || memory.day;
    if (!day || !days.includes(day)) {
      return res.json({ answer: "Please specify a valid day." });
    }
    const availableDoctors = cachedDoctors
      .filter((d) => d[day] === "1")
      .map((d) => `${d.doctorName} (${d.doctorDepartment})`);
    if (availableDoctors.length === 0) {
      return res.json({ answer: `No doctors available on ${capitalize(day)}.` });
    }
    return res.json({ answer: availableDoctors });
  }

  if (intent === "doctor.by_specialty") {
    const specialty =
      entities.find((e) => e.entity === "specialty")?.sourceText?.toLowerCase() || memory.specialty;
    if (!specialty) return res.json({ answer: "Please specify a specialty." });

    const matches = cachedDoctors.filter((d) => {
      const spec = d.specialityTitle?.toLowerCase() || "";
      const dept = d.doctorDepartment?.toLowerCase() || "";
      return spec.includes(specialty) || dept.includes(specialty);
    });

    if (matches.length === 0) {
      return res.json({ answer: `No doctors found for specialty: ${specialty}` });
    }

    if (matches.length === 1) {
      memory.doctor = matches[0].doctorName.split(" (")[0];
      delete memory.doctors;
    } else {
      memory.doctors = matches.map((d) => d.doctorName.split(" (")[0]);
      delete memory.doctor;
    }

    return res.json({
      answer: matches.map((d) => `${d.doctorName} (${d.doctorDepartment})`),
    });
  }

  return res.json({ answer: "Sorry, I didn't understand. Can you rephrase?" });
});

// Session reset
app.post("/reset", (req, res) => {
  const sessionId = req.cookies.sessionId;
  if (sessionId && sessionMemory[sessionId]) {
    delete sessionMemory[sessionId];
  }
  res.clearCookie("sessionId");
  res.json({ success: true });
});

app.listen(4000, () => {
  console.log("✅ NLP expert system running on http://localhost:4000");
});
