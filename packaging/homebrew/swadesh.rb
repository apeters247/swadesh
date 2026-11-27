class Swadesh < Formula
  desc "Bit-exact sacreBLEU BLEU/chrF/chrF++/TER command-line, in Rust"
  homepage "https://github.com/apeters247/swadesh"
  url "https://github.com/apeters247/swadesh/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "06b15207b4eeaf3944c775fdfe93d94651aa2b3273ef26acf2fc15e04b88ceea"
  license "AGPL-3.0-or-later"
  head "https://github.com/apeters247/swadesh.git", branch: "master"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on "rust" => :build

  def install
    # Installs the workspace's CLI member; the binary it puts in place is called `swadesh`.
    system "cargo", "install", *std_cargo_args(path: "crates/swadesh-cli")
  end

  test do
    # The CLI defaults to JSON, as sacreBLEU's does, so the metric name is in the output.
    (testpath/"ref.txt").write("the cat sat on the mat\n")
    (testpath/"hyp.txt").write("the cat is on the mat\n")
    assert_match "\"name\": \"BLEU\"",
                 shell_output("#{bin}/swadesh #{testpath}/ref.txt -i #{testpath}/hyp.txt -m bleu")
    assert_match "swadesh 2.6.0", shell_output("#{bin}/swadesh --version")
  end
end
