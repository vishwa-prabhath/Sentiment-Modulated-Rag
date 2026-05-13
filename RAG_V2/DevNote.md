| **Issues**  |   |   |  
|------------|----|---------------------------------------------------------------|  
| **DATE**   | **ID** | **Description**                                            |  
| 25/2/2025  | 1  | maxtoken length of the model is limited to 512 tokens         |  
| 1/3/2025   | 2  | Discovered that the 512 token length issue arose because we were using the `tiny-dolphin` model, which is relatively very small. |  

| **Solutions**  |   |   |  
|--------------|----|---------------------------------------------------------------|  
| **DATE**    | **ID** | **Description**                                            |  
| 25/2/2025   | 1  | Tokenized output was checked and sliced to 512.               |  
| 1/3/2025    | 2  | Switched to the relatively larger counterpart of it, based on LLaMA 3.2: `cognitivecomputations/Dolphin3.0-Llama3.2-3B`. Also added a truncation system as a fallback. |  

| **TODO**  |  
|----------|  
| Implementing memory chain |