$p = "frontend\src\App.jsx"
$s = [System.IO.File]::ReadAllText($p)

if ($s.Contains("previewInstance")) {
    Write-Host "Preview logic already exists. No changes made."
    exit
}

$old = '  const [studyDetails, setStudyDetails] = useState(null);'
$new = @'
  const [studyDetails, setStudyDetails] = useState(null);
  const [previewInstance, setPreviewInstance] = useState(null);
  const [previewError, setPreviewError] = useState(false);
'@

if (-not $s.Contains($old)) {
    throw "Could not find the studyDetails state. No changes made."
}
$s = $s.Replace($old, $new.TrimEnd())

$marker = '  const details = studyDetails?.MainDicomTags ?? {};'

$effect = @'
  useEffect(() => {
    if (!selectedStudy) {
      setPreviewInstance(null);
      setPreviewError(false);
      return;
    }

    const controller = new AbortController();

    async function loadPreviewInstance() {
      setPreviewInstance(null);
      setPreviewError(false);

      try {
        const response = await fetch(
          `${API_BASE}/orthanc/studies/${encodeURIComponent(selectedStudy)}/series`,
          { signal: controller.signal }
        );

        if (!response.ok) throw new Error("Series request failed");

        const data = await response.json();
        const instanceId = (data.series ?? [])
          .flatMap((series) => series.Instances ?? [])
          .find((id) => typeof id === "string");

        if (!controller.signal.aborted) {
          if (instanceId) setPreviewInstance(instanceId);
          else setPreviewError(true);
        }
      } catch (err) {
        if (err.name !== "AbortError") setPreviewError(true);
      }
    }

    loadPreviewInstance();
    return () => controller.abort();
  }, [selectedStudy]);

'@

if (-not $s.Contains($marker)) {
    throw "Could not find the preview logic insertion point."
}
$s = $s.Replace($marker, $effect + $marker)

$target = '                  <div className="metadata-list">'

$jsx = @'
                  <section className="dicom-preview" aria-label="DICOM image preview">
                    <div className="preview-heading">
                      <strong>Image preview</strong>
                      <span className="metadata-tag">DICOM</span>
                    </div>

                    {previewInstance ? (
                      <img
                        className="dicom-preview-image"
                        src={`${API_BASE}/orthanc/instances/${encodeURIComponent(previewInstance)}/preview`}
                        alt="Preview of the selected DICOM image"
                        onError={() => setPreviewError(true)}
                      />
                    ) : (
                      <div className="preview-placeholder">
                        {previewError
                          ? "No image preview is available for this study."
                          : "Finding an image in the selected study..."}
                      </div>
                    )}

                    <p className="preview-disclaimer">
                      Research preview only. Not for clinical diagnosis.
                    </p>
                  </section>

'@

if (-not $s.Contains($target)) {
    throw "Could not find the metadata section."
}
$s = $s.Replace($target, $jsx + $target)

[System.IO.File]::WriteAllText(
    $p,
    $s,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "DICOM preview added successfully."