# Goal: ISL to ASL Translation Avatar Pipeline

Based on our observations and the challenges faced in this project, we know that predicting physical sign components (HamNoSys tags) directly from video using ML is noisy, inaccurate, and leads to Avatar crashes. 

To achieve cross-regional translation (Indian Sign Language to American Sign Language), we cannot simply translate physical movements. "Hello" in ISL uses completely different handshapes and locations than "Hello" in ASL. Therefore, the pipeline must pass through a **semantic text/gloss layer**.

Here is the proposed architectural plan to pivot the project to an ISL -> ASL Avatar translation system.

## Proposed Architecture

### Phase 1: ISL Video to ISL Gloss (Sign Language Recognition)
**The Challenge:** The current Random Forest models predict granular physical tags (`hamcheek`, `hammovel`), losing the actual meaning of the sign.
**The Solution:** We must replace or retrain the ML pipeline to perform **Action Recognition**. Instead of predicting 10 different physical tags, a deep learning sequence model (like an LSTM, Transformer, or 3D-CNN) will look at the Mediapipe landmarks and classify the entire video into a semantic Gloss (e.g., the text string `"FEVER"` or `"HELLO"`).

### Phase 2: ISL Gloss to ASL Gloss (NLP Translation)
Once we have the meaning in text, we can translate it. 
- **For single words/signs:** The ISL Gloss often maps 1:1 to the ASL Gloss (e.g., `"FEVER"` -> `"FEVER"`).
- **For sentences:** ISL uses Subject-Object-Verb (SOV) grammar, while ASL uses Time-Topic-Comment grammar. We can pass the ISL Gloss through a Large Language Model (like Gemini) or an NLP sequence-to-sequence model to reorganize the sentence into correct ASL grammar.

### Phase 3: ASL Gloss to ASL HamNoSys (The Dictionary Approach)
**The Challenge:** Generating physical XML tags using ML guarantees failure because avatars require perfect grammatical structure, which ML struggles to provide consistently.
**The Solution:** We leverage the **Dictionary Architecture** we just built! We create a `SIGN_DICTIONARY` specifically for ASL.
- The system takes the ASL Gloss (e.g., `"HELLO"`).
- It looks up the hardcoded, perfect HamNoSys reference string for the **American** version of "Hello" (which is a salute from the forehead, unlike the ISL version).
- We use the `hamnosys` PyPI library to instantly convert it to perfect SiGML.

### Phase 4: ASL Avatar Rendering
The perfect, grammatical ASL SiGML is sent to JASigning. The avatar performs the American Sign Language sign perfectly.

---

## User Review Required

> [!IMPORTANT]
> **Major Pivot:** This plan requires abandoning the 10 current `.pkl` classifiers that extract HamNoSys tags, because they do not output the semantic "meaning" (Gloss) of the sign. You will need a Sign Language Recognition (SLR) model that outputs text (e.g., "HELLO") instead of tags.

## Open Questions

1. Do you currently have an ML model or dataset that can predict the **Gloss** (the actual English word) from the ISL video, rather than predicting the physical HamNoSys tags? 
2. If not, would you like me to mock up a placeholder Python script that simulates Phase 1 (so we can test Phase 2 and Phase 3 immediately), or do you want to start building a new ML model using Mediapipe and LSTMs for Phase 1?
