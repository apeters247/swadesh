# Homebrew formula for the standalone CLI.
#
# Lives in a tap: create the repository `apeters247/homebrew-tap` on GitHub, put this file at
# `Formula/swadesh.rb`, and users install with:
#
#     brew install apeters247/tap/swadesh
#
# Before the first push, fill in `sha256` (the tag has to exist first):
#
#     curl -sL https://github.com/apeters247/swadesh/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
#
# Then check it on a machine with Homebrew:
#
#     brew install --build-from-source apeters247/tap/swadesh
#     brew test swadesh
#     brew audit --new --strict swadesh
#     brew style apeters247/tap
#
# This builds from source (about a minute, needs Rust). Once there are prebuilt binaries on the release
# page, switch to the `on_macos`/`on_linux` + `url`/`sha256` form so users do not compile anything.

class Swadesh < Formula
  desc "Bit-exact sacreBLEU BLEU/chrF/chrF++/TER command line, in Rust"
  homepage "https://github.com/apeters247/swadesh"
  url "https://github.com/apeters247/swadesh/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "AGPL-3.0-or-later"
  head "https://github.com/apeters247/swadesh.git", branch: "master"

  depends_on "rust" => :build

  def install
    # The workspace's CLI member; the binary it installs is called `swadesh`.
    system "cargo", "install", *std_cargo_args(path: "crates/swadesh-cli")
  end

  test do
    # The CLI defaults to JSON, as sacreBLEU's does, so the metric name is in the output.
    (testpath/"ref.txt").write("the cat sat on the mat\n")
    (testpath/"hyp.txt").write("the cat is on the mat\n")
    output = shell_output("#{bin}/swadesh #{testpath}/ref.txt -i #{testpath}/hyp.txt -m bleu")
    assert_match "\"name\": \"BLEU\"", output
    assert_match "swadesh 2.6.0", shell_output("#{bin}/swadesh --version")
  end
end