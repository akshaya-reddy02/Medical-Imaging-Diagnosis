$p = "frontend\src\App.jsx"
$s = [System.IO.File]::ReadAllText($p)

if ($s.Contains("previewInstances")) {
    Write-Host "Slice navigation code already exists. No changes made."
    exit
}

Copy-Item $p "$p.before_slices.bak" -Force

$s = $s.Replace(
'  const [previewInstance, setPreviewInstance] = useState(null);',
'  const [previewInstances, setPreviewInstances] = useState([]);' + "`r`n" +
'  const [previewIndex, setPreviewIndex] = useState(0);'
)

$s = $s.Replace(
'      setPreviewInstance(null);',
'      setPreviewInstances([]);' + "`r`n" +
'      setPreviewIndex(0);'
)

$s = $s.Replace(
'      setPreviewInstance(null);',
'      setPreviewInstances([]);' + "`r`n" +
'      setPreviewIndex(0);'
)

$s = $s.Replace(
'        const instanceId = (data.series ?? [])' + "`r`n" +
'          .flatMap((series) => series.Instances ?? [])' + "`r`n" +
'          .find((id) => typeof id === "string");',
'        const instanceIds = [...new Set((data.series ?? [])' + "`r`n" +
'          .flatMap((series) => series.Instances ?? [])' + "`r`n" +
'          .filter((id) => typeof id === "string"))];'
)

$s = $s.Replace(
'          if (instanceId) setPreviewInstance(instanceId);' + "`r`n" +
'          else setPreviewError(true);',
'          if (instanceIds.length > 0) {' + "`r`n" +
'            setPreviewInstances(instanceIds);' + "`r`n" +
'            setPreviewIndex(0);' + "`r`n" +
'          } else {' + "`r`n" +
'            setPreviewError(true);' + "`r`n" +
'          }'
)

$old = @'
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
'@

$new = @'
                    {previewInstances.length > 0 ? (
                      <>
                        <img
                          className="dicom-preview-image"
                          src={`${API_BASE}/orthanc/instances/${encodeURIComponent(previewInstances[previewIndex])}/preview`}
                          alt={`DICOM image ${previewIndex + 1} of ${previewInstances.length}`}
                          onError={() => setPreviewError(true)}
                        />
                        <div className="slice-controls">
                          <button
                            className="secondary-button"
                            disabled={previewIndex === 0}
                            onClick={() => {
                              setPreviewError(false);
                              setPreviewIndex((index) => Math.max(0, index - 1));
                            }}
                          >
                            Previous
                          </button>
                          <span>Image {previewIndex + 1} of {previewInstances.length}</span>
                          <button
                            className="secondary-button"
                            disabled={previewIndex >= previewInstances.length - 1}
                            onClick={() => {
                              setPreviewError(false);
                              setPreviewIndex((index) => Math.min(previewInstances.length - 1, index + 1));
                            }}
                          >
                            Next
                          </button>
                        </div>
                        {previewError && (
                          <p className="preview-disclaimer">
                            This image could not be loaded. Try another image.
                          </p>
                        )}
                      </>
                    ) : (
                      <div className="preview-placeholder">
                        {previewError
                          ? "No image preview is available for this study."
                          : "Finding images in the selected study..."}
                      </div>
                    )}
'@

if (-not $s.Contains($old)) {
    throw "Could not find the existing preview block. The original file is backed up; no update was written."
}

$s = $s.Replace($old, $new)

[System.IO.File]::WriteAllText(
    $p,
    $s,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Slice navigation update completed."