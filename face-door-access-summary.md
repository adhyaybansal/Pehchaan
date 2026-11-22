# Face Recognition Door Access: Discussion Summary

## 1. The idea

A camera is placed above a door. When a person stands outside, the camera detects their face. If the person is on the allowed list, the device tags them, says "<person_name> allowed access", and the door opens.

## 2. Key conclusion: no model training needed

We do **not** need to train a model on our own people. A pre-trained face model already knows how to tell faces apart. We only add people to a database.

| Idea | Verdict |
|---|---|
| Model training | Not needed. Use a pre-trained model. |
| Fine-tuning | Only later, if accuracy is poor in real conditions (dim light, masks, odd camera angle). |
| RAG | Not used. RAG is for LLMs searching text. Face matching uses a similar "search by closeness" idea, but no LLM is involved. |
| In-built service | Yes. Run everything locally on the device (faster, works offline, keeps face data private). |

## 3. How it works (plain words)

1. A face model turns a face into a list of numbers called an **embedding** (a "fingerprint" of the face). AuraFace gives 512 numbers.
2. **Enrollment:** for each allowed person, save their embedding(s) with their name.
3. At the door, the camera finds the face, makes an embedding, and compares it with the saved ones.
4. A very close match = allowed. No match = door stays closed.

## 4. Model choice

**Decision: start with YuNet + AuraFace.** If it feels weak in real tests, then look at other (paid) options.

- **YuNet** is the face finder. It finds faces in the picture and marks 5 key points (both eyes, nose tip, both mouth corners). It is small and fast, runs on a normal CPU, and comes with OpenCV. It does not know who anyone is.
- **AuraFace** is the face identifier. It turns the straightened face crop into 512 numbers for matching.

Pipeline: camera frame -> YuNet finds face -> crop and straighten -> AuraFace makes embedding -> compare with saved list.

### Why not InsightFace?

- InsightFace **code** is MIT licensed (free, including commercial use).
- InsightFace **pre-trained models** (for example `buffalo_l`) are for non-commercial research only. Commercial use needs a license from InsightFace.
- Since this is meant for production and commercial use, we chose models that are free for commercial use.
- In one comparison, AuraFace scored 99.56% accuracy against 99.83% for InsightFace buffalo_l, so the gap is small.

### If we ever need something stronger (paid, on-premise)

Paravision, Innovatrics, Cognitec, Neurotechnology VeriLook, Luxand FaceSDK (from about $499 one-time per platform), NEC (large-scale and government focus). Most prices are on request. The independent benchmark is **NIST FRTE**, so rely on its results rather than vendor "#1" claims. Face++ (Megvii) appeared on a US Entity List in one review, so avoid it for a product we plan to sell.

Design tip: keep the "face -> embedding" step in one small, separate piece of code, so a model can be swapped later without rewriting everything.

## 4b. Anti-spoofing (liveness check)

The models only check **whose face it is**, not whether a live person is standing there. If someone shows a photo or video of an allowed person on a phone, the door could open.

Anti-spoofing adds a check: "is this a real, live face?" Options, from simplest to strongest:

1. A small AI model that spots fakes (glare, pixel patterns, screen edges). Works with a normal camera.
2. Asking the person to blink or turn their head (a video can still fool this).
3. A depth camera (a real face has depth, a screen is flat). Very hard to fool.
4. An infrared (IR) camera.

Plan: start with option 1, and plan for option 3 or 4 in production. For sensitive rooms, consider a second factor (card or PIN).

## 5. Boss's requirements (admin and client screens)

### Choosing the area (4-click box)

- The camera records a full HD frame (stated as 1920x1280; normal Full HD is 1920x1080, so confirm the exact size in camera settings).
- The admin sees the **live full frame** on the admin website.
- In camera settings, the admin clicks **4 points** with the mouse to define the area to cover (bounding box coordinates).
- The camera still takes the full frame, but only this box area is stored or snapped (a crop / ROI).
- Notes: a crop is always a straight rectangle, so use the smallest rectangle covering all 4 points. YuNet runs only inside the crop, which is faster and ignores background people. Click coordinates must match the real video size.

### Tagging people

1. Person arrives. YuNet finds the face and AuraFace makes the embedding.
2. If it matches nobody, the snap appears in the client's screen as "Unknown".
3. The client clicks the snap and types a name (the client tags them as "their person").
4. Next time the person arrives, the system says "<person> access granted".

### Things to decide early

- **One snap is weak.** Save the 3 to 5 best snaps per person for reliable matching.
- **Who may tag?** If any client user can tag anyone, a mistake or bad actor can open the door for a stranger. Consider roles (admin only, or approval) and a log of who tagged whom.
- **Consent.** People should agree to their face being stored (India's DPDP Act applies to face data).

## 6. Hardware and safety notes

- A relay board (Raspberry Pi or ESP32) unlocks the door.
- Set a strict match threshold, and log every attempt with time and photo.
- Keep a backup way to open the door (keypad, card, or manual key).
- Store embeddings securely.

## 7. Build order (small steps)

1. Python installed on the laptop **(current step)**
2. Detect a face from the webcam with YuNet
3. Make the embedding with AuraFace and match two photos of the same person
4. Add the crop box (a fixed one first, then the 4-click version in the website)
5. Unknown snaps screen and tagging
6. Door relay and anti-spoofing

## 8. Next action

Check Python on the laptop. Open a terminal (Windows: Command Prompt or PowerShell) and run:

```
python --version
```

On Mac or Linux, try `python3 --version` if the first one does not work. We want Python 3.10 or higher (3.11 or 3.12 is best).
