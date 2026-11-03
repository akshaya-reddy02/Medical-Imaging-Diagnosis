import asyncio
import logging

import httpx
import torch
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class OrthancMonitor:
    def __init__(
        self,
        base_url,
        username,
        password,
        classifier,
        image_transform,
        threshold,
        poll_interval=5,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.classifier = classifier
        self.image_transform = image_transform
        self.threshold = threshold
        self.poll_interval = poll_interval
        self.last_sequence = 0
        self.processed_studies = set()
        self.running = False

    async def get_json(self, client, path):
        response = await client.get(path)
        response.raise_for_status()
        return response.json()

    async def process_study(self, client, study_id):
        if study_id in self.processed_studies:
            return

        try:
            study = await self.get_json(
                client, f"/studies/{study_id}"
            )

            instances = study.get("Instances", [])
            if not instances:
                return

            instance_id = instances[0]

            response = await client.get(
                f"/instances/{instance_id}/preview"
            )
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as source:
                image = source.convert("RGB")

            tensor = self.image_transform(image).unsqueeze(0)

            with torch.inference_mode():
                probability = torch.sigmoid(
                    self.classifier(tensor)
                ).item()

            result = {
                "study_id": study_id,
                "target": "Cardiomegaly",
                "probability": round(float(probability), 6),
                "threshold": self.threshold,
                "predicted_label": int(
                    probability >= self.threshold
                ),
                "clinical_use": False,
            }

            logger.info(
                "Automatic Orthanc inference result: %s", result
            )

            self.processed_studies.add(study_id)

        except Exception:
            logger.exception(
                "Could not process Orthanc study %s", study_id
            )

    async def poll_once(self, client):
        response = await client.get(
            "/changes",
            params={
                "last": self.last_sequence,
                "limit": 100,
            },
        )
        response.raise_for_status()
        data = response.json()

        for change in data.get("Changes", []):
            sequence = int(change.get("Seq", 0))
            self.last_sequence = max(
                self.last_sequence, sequence
            )

            if (
                change.get("ChangeType") == "StableStudy"
                and change.get("ResourceType") == "Study"
            ):
                await self.process_study(
                    client, change["ID"]
                )

    async def run(self):
        self.running = True

        async with httpx.AsyncClient(
            base_url=self.base_url,
            auth=self.auth,
            timeout=20,
            trust_env=False,
        ) as client:
            while self.running:
                try:
                    await self.poll_once(client)
                except Exception:
                    logger.exception(
                        "Orthanc monitoring poll failed."
                    )

                await asyncio.sleep(self.poll_interval)

    def stop(self):
        self.running = False