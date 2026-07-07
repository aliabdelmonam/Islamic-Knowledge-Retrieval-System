import os
import requests

os.makedirs("Physics_PDFs", exist_ok=True)

session = requests.Session()

for i in range(1, 11):
    urls = [
        # Primary
        f"https://ellibrary.moe.gov.eg/sec3guideforms/publish{i}/Physics_EN_Model_{i}.pdf",

        # Fallback
        f"https://elearnningcontent.blob.core.windows.net/"
        f"elearnningcontent/2026/Secondry/Secondry3/Term1/"
        f"Guidance_models_2025_2026/Physics_EN_3_Secondary_{i}.pdf",
    ]

    filename = f"Physics_PDFs/Physics_{i}.pdf"

    for url in urls:
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f"✓ Downloaded {i} from {url}")
                break
            else:
                print(f"Failed ({r.status_code}): {url}")
        except Exception as e:
            print(f"Error: {url} -> {e}")
    else:
        print(f"✗ Could not download file {i}")