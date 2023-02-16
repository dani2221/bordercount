using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace borderCounterApi.Migrations
{
    /// <inheritdoc />
    public partial class prevMins : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "PrevMinutes",
                table: "BorderInformationIntervals",
                type: "integer",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "PrevMinutes",
                table: "BorderInformationIntervals");
        }
    }
}
