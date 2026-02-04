# Kullanıcıdan commit mesajı iste
$msg = Read-Host "Commit mesajını gir (boş bırakırsan 'Guncelleme' olacak)"
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Guncelleme" }

# Proje klasörüne git
Set-Location "C:\Users\hp\OneDrive\Masaüstü\deprem-projesi"

# Git işlemleri
git add .
git commit -m "$msg"
git push
