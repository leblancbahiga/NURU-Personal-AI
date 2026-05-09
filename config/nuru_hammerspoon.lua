-- NURU — Raccourci clavier Option+Espace
-- Ouvre l'interface NURU (overlay natif PySide6)
-- Utilise le venv Python où PySide6 est installé

local nuru_dir = os.getenv("HOME") .. "/Downloads/Assistant IA"
local venv_python = nuru_dir .. "/.venv/bin/python3"
local nuru_overlay = nuru_dir .. "/src/nuru_overlay.py"

function toggleNuru()
  -- Vérifier si PySide6 est dispo dans le venv
  local handle = io.popen(venv_python .. " -c \"from PySide6.QtWidgets import QApplication; print('ok')\" 2>/dev/null")
  local has_pyside = handle:read("*a"):match("ok")
  handle:close()

  if has_pyside then
    -- Lancer l'overlay holographique natif via le venv
    local overlay_pid = io.popen("pgrep -f nuru_overlay.py 2>/dev/null"):read("*a")
    if overlay_pid == "" then
      hs.task.new(venv_python, function() end, {nuru_overlay}):start()
    else
      hs.notify.new({title="NURU", informativeText="Déjà ouvert"}):send()
    end
  else
    hs.notify.new({title="NURU", informativeText="PySide6 introuvable. Lance d'abord deploy_nuru.command"}):send()
  end
end

-- Option+Space (⌥+␣)
hs.hotkey.bind({"alt"}, "space", toggleNuru)

hs.notify.new({title="NURU", informativeText="Prêt — ⌥+Espace pour parler à NURU"}):send()
